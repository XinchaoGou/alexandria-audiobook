"""Regression tests for preserving primary WAV writer failures."""
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import cli


class WavMergeErrorTests(unittest.TestCase):
    def assert_writer_failure(self, primary, close_error, method="writeframes"):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                handle.writeframes(b"\x01\x02" * 8)
            real_open = wave.open
            writer = mock.Mock()
            writer.__enter__ = mock.Mock(return_value=writer)
            writer.__exit__ = mock.Mock(side_effect=close_error)
            writer.close.side_effect = close_error
            if primary is not None:
                getattr(writer, method).side_effect = primary
                # Both buffered and streaming writers must preserve the same fault.
                if method == "writeframes":
                    writer.writeframesraw.side_effect = primary
            def open_wave(path, mode):
                return writer if mode == "wb" else real_open(path, mode)
            expected = primary if primary is not None else close_error
            with mock.patch.object(cli.wave, "open", side_effect=open_wave):
                with self.assertRaises(BaseException) as raised:
                    cli.BookPipeline._merge_wav_files(
                        [source], ["a"], [], 500, 250, Path(directory) / "out.wav"
                    )
            self.assertIs(raised.exception, expected)

    def test_write_error_survives_close_error(self):
        self.assert_writer_failure(OSError("write fault"), OSError("close fault"))

    def test_interruption_survives_close_error(self):
        self.assert_writer_failure(KeyboardInterrupt("interrupted"), OSError("close fault"))

    def test_setup_error_survives_close_interruption(self):
        self.assert_writer_failure(ValueError("setup fault"), KeyboardInterrupt(), "setparams")

    def test_successful_body_close_error_is_fatal(self):
        self.assert_writer_failure(None, OSError("close fault"))


if __name__ == "__main__":
    unittest.main()
