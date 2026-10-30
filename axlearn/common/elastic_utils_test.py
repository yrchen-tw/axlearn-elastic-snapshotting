# Copyright © 2025 Apple Inc.

"""Tests elastic_utils."""

from unittest import mock

from absl.testing import absltest

from axlearn.common import elastic_utils


class ElasticUtilsTest(absltest.TestCase):
    def tearDown(self):
        elastic_utils.elastic_manager = None
        super().tearDown()

    def test_total_slice_count(self):
        self.assertIsNone(elastic_utils.total_slice_count())
        elastic_utils.elastic_manager = mock.MagicMock(total_slice_count=4)
        self.assertEqual(elastic_utils.total_slice_count(), 4)

    def test_available_slice_count(self):
        self.assertIsNone(elastic_utils.available_slice_count())
        mgr = elastic_utils.elastic_manager = mock.MagicMock(total_slice_count=4)
        mgr.slice_to_devices = {i: [] for i in range(4)}
        with mock.patch.object(elastic_utils, "elastic") as mock_elastic:
            mock_elastic.get_active_slice_indices.return_value = {0, 2, 3}
            self.assertEqual(elastic_utils.available_slice_count(), 3)
            mock_elastic.get_active_slice_indices.assert_called_once_with(
                slice_to_devices=mgr.slice_to_devices
            )
            # A failing health check must not interrupt the elastic recovery.
            mock_elastic.get_active_slice_indices.side_effect = RuntimeError("unreachable")
            self.assertIsNone(elastic_utils.available_slice_count())


if __name__ == "__main__":
    absltest.main()
