import unittest

import numpy as np

import compressor
from main import detail_to_max_error


class ApproximationTests(unittest.TestCase):
    def test_detail_mapping_has_expected_range_and_direction(self):
        self.assertAlmostEqual(detail_to_max_error(0), 0.30)
        self.assertAlmostEqual(detail_to_max_error(50), 0.03)
        self.assertAlmostEqual(detail_to_max_error(100), 0.003)
        self.assertGreater(detail_to_max_error(20), detail_to_max_error(80))

    def test_detail_mapping_rejects_out_of_range_values(self):
        for detail in (-1, 101, 50.5, True):
            with self.subTest(detail=detail), self.assertRaises(ValueError):
                detail_to_max_error(detail)

    def test_error_is_normalized_mean_absolute_error(self):
        original = np.zeros((2, 2, 3), dtype=np.uint8)
        approximation = np.full_like(original, 255)
        self.assertEqual(compressor.mean_absolute_error(original, approximation), 1.0)

    def test_zero_error_reproduces_input(self):
        image = np.array(
            [[0, 10, 20], [30, 40, 50]], dtype=np.uint8
        )
        np.testing.assert_array_equal(compressor.process_image(image, 0), image)

    def test_uniform_image_needs_no_visible_approximation(self):
        image = np.full((5, 7, 3), (12, 90, 210), dtype=np.uint8)
        result = compressor.process_image(image, 0.05)
        np.testing.assert_array_equal(result, image)

    def test_stricter_target_does_not_increase_measured_error(self):
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape((8, 8, 3))
        loose = compressor.process_image(image, 0.2)
        strict = compressor.process_image(image, 0.02)
        self.assertLessEqual(
            compressor.mean_absolute_error(image, strict),
            compressor.mean_absolute_error(image, loose),
        )

    def test_supports_rgb_and_rgba(self):
        for channels in (3, 4):
            image = np.arange(6 * 5 * channels, dtype=np.uint8).reshape((6, 5, channels))
            result = compressor.process_image(image, 0.1)
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, image.dtype)

    def test_rejects_invalid_error_target(self):
        image = np.zeros((2, 2), dtype=np.uint8)
        for target in (-0.1, 1.1, float("nan"), "0.1"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                compressor.process_image(image, target)

    def test_rejects_empty_and_out_of_range_images(self):
        with self.assertRaises(ValueError):
            compressor.process_image(np.empty((0, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            compressor.process_image(np.array([[256]], dtype=np.uint16))
        with self.assertRaises(ValueError):
            compressor.process_image(np.array([[complex(1, 2)]]))
        with self.assertRaises(ValueError):
            compressor.process_image(np.array([[np.nan]]))


if __name__ == "__main__":
    unittest.main()
