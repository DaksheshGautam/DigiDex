import base64
import io
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from app import app, model

class AppTests(unittest.TestCase):
    def test_export_matches_colab(self):
        data = np.load(Path(__file__).parent / 'models/verification.npz')
        result = model.predict(data['images'])
        np.testing.assert_allclose(result, data['probabilities'], atol=1e-5, rtol=1e-4)
        np.testing.assert_array_equal(result.argmax(1), data['probabilities'].argmax(1))

    def test_web_prediction_and_empty_canvas(self):
        client = app.test_client()
        self.assertEqual(client.get('/healthz').status_code, 200)
        self.assertIn(b'98.86%', client.get('/').data)
        data = np.load(Path(__file__).parent / 'models/verification.npz')
        image = Image.fromarray((data['images'][0,:,:,0]*255).astype('uint8')).resize((280,280))
        def payload(image):
            buf = io.BytesIO(); image.save(buf, format='PNG')
            return {'image': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()}
        result = client.post('/predict', json=payload(image))
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json['digit'], int(data['labels'][0]))
        self.assertEqual(len(result.json['probabilities']), 10)
        self.assertEqual(client.post('/predict', json=payload(Image.new('L',(280,280)))).status_code, 400)
        self.assertEqual(client.post('/predict', json={'image':'bad'}).status_code, 400)

if __name__ == '__main__':
    unittest.main()
