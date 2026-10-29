# Deferred AUDIO draft

`AudioRef.from_asset` captures bounded immutable encoded bytes from an authorized
file asset. It returns AUDIO without decoding. Existing `from_value`, eager
audio dictionaries and provider registrations keep their behavior.

```python
from comfy_api.latest import io, sdk

asset = await sdk.ctx().assets.resolve("input", name)
audio = await sdk.AudioRef.from_asset(
    asset, max_source_bytes=64 * 1024 * 1024,
    max_decoded_bytes=128 * 1024 * 1024, timeout_seconds=30,
    start_seconds=1.0, duration_seconds=None,
)
return io.NodeOutput(audio)
```

The closed `stream0_pcm_f32_v1` policy selects the first audio stream at its
native rate and channel layout, converts to planar float32 and returns
`waveform[batch, channels, samples]` with `sample_rate`. Window boundaries floor
seconds to sample indices measured from the first decoded sample. `None` reads
until EOF under the decoded-byte and cooperative time budgets. This policy is
not a claim of sample equivalence with another decoder.

Internal reference resolution, output unwrapping, edge transfer and kind
recognition do not consume audio. `await audio.value()`, raw materialization,
ordinary mapping lookup (including `sample_rate`), iteration and length do.
The native video output consumer realizes audio at actual use. Ordinary native
AUDIO consumers receive a managed mapping, not a recipe dictionary. A consumer
requiring an exact `dict` needs an explicit realization adapter; representative
Core and corpus consumers still require qualification.

Success caches one value. Concurrent consumers share one attempt and its error;
a later access may retry a failed attempt. At most 16 waiters join one value.
Cancelling a waiter does not cancel or unaccount work another consumer owns.
An output/cache owns its immutable source independently of the producer's
execution table; no producer process or scratch pathname must survive.

`AudioDecodeError` identifies invalid encoded data, unexpected container EOF or
absence of an audio stream. Only PyAV `InvalidDataError` and `EOFError` are
normalized; other PyAV failures keep their error types.
The matched Guest resolver restores that public exception from the broker's
structured error type for AUDIO materialization. Permission, transport,
cancellation, dependency, deadline and resource-limit failures retain their
own errors. A pack may catch a decode failure at its original catch site;
catching every exception would also hide failures that are not codec failures.
This discriminator and the wire path remain unqualified test sources.

The native pool bounds 16 live values and 1 GiB of conservatively reserved payload
memory. Each reservation includes its requested source bound and twice its
decoded bound. This accounts for retained/in-flight payloads, not codec internals
or independently retained raw buffers. The pool stores weak metadata, not audio.
RAM-cache sizing is non-consuming; external serialization is refused before a
provider receives outputs containing this value. External cache serialization of
deferred AUDIO is unavailable in this version.

Decode deadlines are cooperative at frame boundaries. They cannot preempt a
blocked codec, native mapping consumer or filesystem syscall. They are not a
process-memory or hard execution deadline. A separately selected supervising
runtime must qualify and enforce those boundaries where required.

Capture requires no-follow regular-file support. Unsupported platforms fail
explicitly before capture; a Windows implementation and qualification remain
open. There is no pathname fallback.

Trusted native provider composition explicitly opts into the new operation.
Older `InProcessOps` subclasses keep it disabled. The matched Guest bridge must
require assets permission and finite operator-selected source/decoded/time
limits before capture; raw permission is still required before buffers return to
a Guest. An AUDIO handle alone does not grant raw buffers. The bridge rechecks
dispatch identity after asynchronous realization. No new method is required of
the optional runtime-checkable `AssetMediaProvider`.

This source draft is unqualified. Imports, generated declarations, tests,
decoder fidelity, ordinary local ComfyUI, Windows/Electron and Guest checks are
UNRUN. Corpus adoption requires the bound caller census, compatibility fixes,
canonical declarations, affected manifest/patch regeneration and roundtrip and
cross-pack results in a separately selected slot.
