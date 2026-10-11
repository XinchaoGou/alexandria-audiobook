"""Synthetic PCM checks; no TTS, models, private inputs or optional packages."""
import tempfile
import tracemalloc
import unittest
import wave
from pathlib import Path
from unittest import mock

import cli


def write_wav(path, data, channels=1, width=2, rate=8000):
    with wave.open(str(path), "wb") as handle:
        handle.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        handle.writeframes(data)


class StreamingWavMergeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "output.wav"

    def merge(self, paths, speakers=None, pauses=None, output=None):
        return cli.BookPipeline._merge_wav_files(
            paths, speakers if speakers is not None else ["a"] * len(paths),
            pauses if pauses is not None else [], 31, 17, output or self.output
        )

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob(".merge-wav-*")), [])

    def test_exact_format_pcm_and_rounded_timeline(self):
        for channels in (1, 2):
            for width in (1, 2, 3, 4):
                for rate in (8000, 11025, 44100, 48000):
                    with self.subTest(channels=channels, width=width, rate=rate):
                        counts = [0, 1, 16385, 7, 0]
                        payloads = [bytes([index + 1]) * count * channels * width
                                    for index, count in enumerate(counts)]
                        paths = [self.root / f"{i}.wav" for i in range(5)]
                        for path, data in zip(paths, payloads):
                            write_wav(path, data, channels, width, rate)
                        speakers = ["a", "a", "b", "b", "a"]
                        pauses = [None, "3", -4, None]
                        gaps = [int(gap * rate / 1000) for gap in (17, 3, 0, 31)]
                        expected_pcm = payloads[0]
                        expected_timeline = [{"line_id": 1, "start_ms": 0, "end_ms": 0}]
                        cursor = 0
                        for index, data in enumerate(payloads[1:], 1):
                            cursor += gaps[index - 1]
                            start = round(cursor * 1000 / rate)
                            cursor += counts[index]
                            expected_timeline.append({"line_id": index + 1,
                                "start_ms": start, "end_ms": round(cursor * 1000 / rate)})
                            expected_pcm += b"\0" * gaps[index - 1] * channels * width + data
                        expected = self.root / "expected.wav"
                        write_wav(expected, expected_pcm, channels, width, rate)
                        self.assertEqual(self.merge(paths, speakers, pauses), expected_timeline)
                        self.assertEqual(self.output.read_bytes(), expected.read_bytes())

    def test_many_lines_and_single_empty_line(self):
        path = self.root / "line.wav"
        write_wav(path, b"\x01\x02")
        timeline = self.merge([path] * 2000, pauses=[0] * 1999)
        self.assertEqual(len(timeline), 2000)
        self.assertEqual(timeline[-1], {"line_id": 2000, "start_ms": 250, "end_ms": 250})
        with wave.open(str(self.output)) as handle:
            self.assertEqual(handle.getnframes(), 2000)
        write_wav(path, b"")
        self.assertEqual(self.merge([path], []), [{"line_id": 1, "start_ms": 0, "end_ms": 0}])
        with wave.open(str(self.output)) as handle:
            self.assertEqual(handle.getnframes(), 0)

    def test_wide_frames_preserve_format_without_large_blocks(self):
        source = self.root / "wide.wav"
        data = b"\x01" * 7000 * 4 * 6
        write_wav(source, data, channels=7000, width=4)
        self.merge([source])
        self.assertEqual(self.output.read_bytes(), source.read_bytes())

    def test_truncated_inputs_keep_available_pcm_and_declared_timestamps(self):
        for remove in (2, 3, 40):
            for position in (0, 1):
                with self.subTest(remove=remove, position=position):
                    paths = [self.root / "a.wav", self.root / "b.wav"]
                    for path in paths:
                        write_wav(path, b"\x01\x02" * 20)
                    path = paths[position]
                    path.write_bytes(path.read_bytes()[:-remove])
                    expected_pcm = (b"\x01\x02" * 20)[:40-remove] if position == 0 else b"\x01\x02" * 20
                    expected_pcm += b"\0" * (17 * 8 * 2)
                    expected_pcm += (b"\x01\x02" * 20)[:40-remove] if position == 1 else b"\x01\x02" * 20
                    expected = self.root / "expected.wav"
                    write_wav(expected, expected_pcm)
                    self.assertEqual(self.merge(paths), [
                        {"line_id": 1, "start_ms": 0, "end_ms": 2},
                        {"line_id": 2, "start_ms": 20, "end_ms": 22},
                    ])
                    self.assertEqual(self.output.read_bytes(), expected.read_bytes())

    def test_missing_or_corrupt_late_input_preserves_prior_output(self):
        first = self.root / "a.wav"
        write_wav(first, b"\x01\x02")
        bad = self.root / "bad.wav"
        for data in (None, b"invalid WAV data"):
            if data is not None:
                bad.write_bytes(data)
            self.output.write_bytes(b"previous output")
            with self.assertRaises(cli.PipelineError) as raised:
                self.merge([first, bad])
            self.assertIn("无法读取音频行", str(raised.exception))
            self.assertEqual(self.output.read_bytes(), b"previous output")
            self.assert_no_staging()

    def test_format_mismatches_preserve_prior_output(self):
        first = self.root / "a.wav"
        bad = self.root / "bad.wav"
        write_wav(first, b"\x01\x02")
        for channels, width, rate in ((2, 2, 8000), (1, 1, 8000), (1, 2, 44100)):
            write_wav(bad, b"\0" * channels * width, channels, width, rate)
            self.output.write_bytes(b"previous output")
            with self.assertRaisesRegex(cli.PipelineError, "音频行参数不一致"):
                self.merge([first, bad])
            self.assertEqual(self.output.read_bytes(), b"previous output")
            self.assert_no_staging()

    def test_no_inputs_or_missing_first(self):
        self.output.write_bytes(b"previous output")
        for paths in ([], [self.root / "missing.wav"]):
            with self.assertRaises(cli.PipelineError):
                self.merge(paths)
            self.assertEqual(self.output.read_bytes(), b"previous output")

    def test_reads_writes_and_memory_are_bounded(self):
        source = self.root / "large.wav"
        write_wav(source, b"\x01\x02" * (4 * 1024 * 1024))
        real_open = wave.open
        reads, writes = [], []
        def open_wave(path, mode):
            handle = real_open(path, mode)
            if mode == "rb":
                real_read = handle.readframes
                def read(count):
                    reads.append(count)
                    return real_read(count)
                handle.readframes = read
            else:
                real_write = handle.writeframesraw
                def write(data):
                    writes.append(len(data))
                    return real_write(data)
                handle.writeframesraw = write
            return handle
        tracemalloc.start()
        try:
            with mock.patch.object(cli.wave, "open", side_effect=open_wave):
                self.merge([source, source], pauses=[100000])
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLessEqual(max(reads), 16384)
        self.assertLessEqual(max(writes), 32768)
        self.assertLess(peak, 1024 * 1024)
        with wave.open(str(self.output)) as handle:
            self.assertEqual(handle.getnframes(), 8 * 1024 * 1024 + 800000)

    def test_write_close_and_interruptions_preserve_prior_output(self):
        source = self.root / "a.wav"
        write_wav(source, b"\x01\x02")
        real_open = wave.open
        for primary, secondary in ((OSError("write"), OSError("close")),
                (KeyboardInterrupt(), OSError("close")),
                (None, OSError("close"))):
            with self.subTest(primary=primary):
                self.output.write_bytes(b"previous output")
                def open_wave(path, mode):
                    handle = real_open(path, mode)
                    if mode == "wb":
                        real_close = handle.close
                        def close():
                            real_close()
                            handle.close = real_close
                            raise secondary
                        handle.close = close
                        if primary is not None:
                            handle.writeframesraw = mock.Mock(side_effect=primary)
                    return handle
                with mock.patch.object(cli.wave, "open", side_effect=open_wave):
                    with self.assertRaises(BaseException) as raised:
                        self.merge([source])
                self.assertIs(raised.exception, primary if primary is not None else secondary)
                self.assertEqual(self.output.read_bytes(), b"previous output")
                self.assert_no_staging()

    def test_publish_failure_preserves_output_and_cleans_staging(self):
        source = self.root / "a.wav"
        write_wav(source, b"\x01\x02")
        self.output.write_bytes(b"previous output")
        with mock.patch.object(cli.os, "replace", side_effect=OSError("publish")):
            with self.assertRaisesRegex(OSError, "publish"):
                self.merge([source])
        self.assertEqual(self.output.read_bytes(), b"previous output")
        self.assert_no_staging()

    def test_read_fault_after_first_block_preserves_output(self):
        source = self.root / "a.wav"
        write_wav(source, b"\x01\x02" * 20000)
        real_open = wave.open
        for position in (0, 1):
            calls = 0
            def open_wave(path, mode):
                nonlocal calls
                handle = real_open(path, mode)
                if mode == "rb":
                    calls += 1
                    # First open reads metadata; subsequent opens copy PCM.
                    if calls == position + 2:
                        real_read = handle.readframes
                        handle.readframes = mock.Mock(side_effect=[
                            real_read(16384), OSError("read fault")
                        ])
                return handle
            self.output.write_bytes(b"previous output")
            with mock.patch.object(cli.wave, "open", side_effect=open_wave):
                with self.assertRaises(cli.PipelineError) as raised:
                    self.merge([source, source])
            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertEqual(self.output.read_bytes(), b"previous output")
            self.assert_no_staging()


if __name__ == "__main__":
    unittest.main()
