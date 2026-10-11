# WAV writer error preservation

`BookPipeline._merge_wav_files` preserves the original setup, write, or
interruption exception when closing its WAV writer also fails. A close error
after a successful body still fails the operation. This change does not add
atomic publication to the buffered writer.

Adapted from Michii's MIT-licensed
[Alexandria Audiobook 2 #1095](https://github.com/on22s/alexandria-audiobook2/pull/1095),
commit `4cd44920622b96c64f8b44edd6ddd9891107b3ee`, also ported in
[Alexandria Audiobook Nya #30](https://github.com/on22s/alexandria-audiobook-nya/pull/30).
The existing repository MIT license and Copyright (c) 2026 Finrandojin notice
cover the donor code and remain intact.
