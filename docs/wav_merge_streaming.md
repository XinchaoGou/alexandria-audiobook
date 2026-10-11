# Bounded dialogue WAV merging

`BookPipeline._merge_wav_files` copies PCM and emits zero-byte silence in
blocks of at most 16,384 frames and 64 KiB (or one frame if wider). The first
input determines the output format.
It retains format rejection, pause override rules, integer gap frame
truncation, and rounded SFX timestamps. No trimming or resampling occurs here;
the later 48 kHz mono PCM16 mix and 44.1 kHz mono AAC-LC 96 kbps publication
policy are unchanged.

Input WAVs with a valid header but truncated PCM remain accepted, as before:
available bytes are copied, while the timeline uses declared frame counts.
This includes an incomplete final PCM frame. Consequently, a truncated input
can still make the timeline longer than the actual output. Rejecting truncated
WAVs would be a separate behavior change.

Output is staged in a unique owned directory beside the destination, then
atomically replaced only after the WAV writer closes successfully. Late input,
write, interruption, close, and publication failures leave prior output intact
and remove the owned staging directory. Primary failures survive secondary
writer close errors; a close error after a successful body remains fatal.

PCM storage is bounded by the chosen format and block size; the returned
timeline still grows with line count. The original algorithm was already
linear in PCM size. This change reduces memory use and makes no quadratic
speedup claim. It adds temporary disk use for one complete output and replaces
the destination inode on success (including replacing a destination symlink).
It does not add RF64 support, snapshot concurrently changing input files, or
provide durability against power loss.

Adapted for stdlib `wave` from Michii's MIT-licensed
[Alexandria Audiobook 2 #1086](https://github.com/on22s/alexandria-audiobook2/pull/1086),
commit `5f0b132cd8b4dde16e949688bf85db1eb6c5e39e`, and
[Alexandria Audiobook Nya #28](https://github.com/on22s/alexandria-audiobook-nya/pull/28).
Error cleanup follows
[AA2 #1095](https://github.com/on22s/alexandria-audiobook2/pull/1095).
The existing repository MIT license and Copyright (c) 2026 Finrandojin notice
cover the donor code and remain intact. The donor Pydub helper, its layout
restrictions, and its silence resampling were not copied.
