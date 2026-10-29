# Encoded media groups draft

`ctx.output.reserve_media_group` returns a dispatch-owned `MediaGroupRef`.
It reserves a common basename/counter beneath `.comfy-media` under output or
temp. Pack code supplies encoded bytes; this facility does not select encoders,
convert pixels or generate metadata.

```python
async with await sdk.ctx().output.reserve_media_group(
    "video/example", max_bytes=128 * 1024 * 1024, max_entries=8,
    timeout_seconds=30, failed_entry_policy="retain", save_output=True,
) as group:
    await group.reserve("utility", ".png", "utility")
    await group.publish("utility", encoded_png_chunks(), max_bytes=4 * 1024 * 1024)
    records = await group.entries()
```

An entry ID declares one suffix and utility/intermediate/final role. Reservation
creates a logical name without a readable asset or preview. Actual generated
names must fit the existing255-byte component and1024-byte logical-name bounds;
case-folded duplicate suffixes are refused. Codec validation happens at
publication, so creating a group does not move later format failures ahead of
earlier completed files.

Publication admits bounded1..1 MiB byte chunks for closed PNG/GIF/WebP/MP4/MOV/
WebM/Matroska extensions and checks their container signatures. This is not a
decoder, full container validator or preview grant. Existing authorized asset
and preview routes must separately validate a committed file. Source bytes and
metadata remain unchanged.

Each completed file appears without overwrite. Earlier completed entries remain
if a later entry fails. The required `failed_entry_policy` chooses retain or
discard for partial bytes from a failed publication. Retention and cleanup
failures are recorded separately; they do not replace an existing primary
encoder/cancellation error. A cleanup error after successful publication can
raise even though the descriptor records a committed file. A committed entry ID
replay returns its existing descriptor without consuming a replacement stream.
It is not a content-update operation.

Failure byte counts report completed writes or a subsequently observed file
size. If file size cannot be read, the remaining byte reservation stays charged
conservatively without presenting that bound as observed bytes. Cleanup and
retention errors also reach the owner log after a dispatch revokes its refs.

`entries()` distinguishes reserved, publishing, committed, failed and deleted
states, bytes written and actual materialization. Authorized snapshots remain
available after the lease deadline until the dispatch/ref retires. New writes
remain refused after deadline. `remove()` deletes only an owned, unchanged
published inode while preserving its logical name in the record. This supports
consumers that list an intermediate after deleting it.

The creating ref table owns group cleanup. Explicit close/release, native table
retirement and matched Guest completion/kill revoke the group. Group handles
cannot be returned directly, nested or through structured ValueRefs. Pending
publication remains owned until cancellation/iterator cleanup unwinds. Native
iterator cleanup and deadlines are cooperative and cannot stop uncooperative
native Python code. Closed-event-loop teardown remains a qualification boundary.
No external file handle may be justified by a garbage-collection promise.

Limits are 16 successful group reservations per creating output-context
instance (normally one Core execution), 256 entries per group, 1 GiB per group
and at most 120 seconds. A reused output context conservatively accumulates its
quota across Guest calls; closing a group does not reset it. Guest requests also require output permission and the
operator's selected finite byte/entry/time caps; defaults deny the new verbs.
Existing `save_binary` extensions, behavior and limits remain unchanged.

The reserved namespace is fenced at standard managed prefix, explicit filename,
text and copy writers, including resolved path aliases. Direct privileged native
filesystem writers and arbitrary custom trusted providers are outside this
guarantee. Paths/counters differ from legacy direct output naming. Exact legacy
names require an explicitly admitted compatibility change or a separately
reviewed allocator integration; this draft does not claim naming parity.

Descriptor-relative/no-follow filesystem support is required. Other platforms
fail explicitly before creating directories. Windows support remains open; an
unsafe pathname fallback is not provided.

These are additive draft operations, not a complete pack conversion. Original
failed-file inventories, encoder faults and pack-specific catch/removal order
remain independent oracle cases. Partial Pillow files are unobserved. No imports,
generated declarations, tests, actual codecs, ordinary local ComfyUI or Guest
qualification ran. Canonical contract generation and affected corpus manifest/
patch regeneration, roundtrips and cross-pack tests are adoption gates.
