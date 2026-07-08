"""Unit tests for comfy_execution/jobs.py"""

import pytest

from comfy_execution.jobs import (
    JobStatus,
    is_previewable,
    normalize_queue_item,
    normalize_history_item,
    normalize_output_item,
    normalize_outputs,
    get_outputs_summary,
    apply_sorting,
    has_3d_extension,
    validate_job_id,
)


class TestValidateJobId:
    """validate_job_id guards job creation: POST /prompt rejects ids it raises on."""

    def test_canonical_form_passes_through(self):
        cid = "a1b2c3d4-e5f6-7a89-b0c1-d2e3f4a5b6c7"
        assert validate_job_id(cid) == cid

    @pytest.mark.parametrize(
        "variant",
        [
            "A1B2C3D4-E5F6-7A89-B0C1-D2E3F4A5B6C7",          # uppercase
            "{a1b2c3d4-e5f6-7a89-b0c1-d2e3f4a5b6c7}",        # braced
            "urn:uuid:a1b2c3d4-e5f6-7a89-b0c1-d2e3f4a5b6c7",  # URN
            "a1b2c3d4e5f67a89b0c1d2e3f4a5b6c7",              # bare hex
            " a1b2c3d4-e5f6-7a89-b0c1-d2e3f4a5b6c7 ",        # padded
        ],
    )
    def test_non_canonical_spellings_rejected(self, variant):
        # uuid.UUID parses all of these, but accepting them would silently
        # rewrite the client's id (history keys, websocket events, and
        # /interrupt matching all match the stored form exactly).
        with pytest.raises(ValueError):
            validate_job_id(variant)

    @pytest.mark.parametrize(
        "bad",
        ["", "not-a-uuid", "prompt-123", "a1b2c3d4-e5f6-7a89-b0c1", "None"],
    )
    def test_non_uuid_strings_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_job_id(bad)

    @pytest.mark.parametrize("bad", [123, 1.5, True, None, ["a"], {"id": "x"}])
    def test_non_strings_rejected(self, bad):
        # uuid.UUID raises AttributeError/TypeError on non-strings; the helper
        # must normalize those to ValueError so callers need one except clause.
        with pytest.raises(ValueError):
            validate_job_id(bad)


class TestJobStatus:
    """Test JobStatus constants."""

    def test_status_values(self):
        """Status constants should have expected string values."""
        assert JobStatus.PENDING == 'pending'
        assert JobStatus.IN_PROGRESS == 'in_progress'
        assert JobStatus.COMPLETED == 'completed'
        assert JobStatus.FAILED == 'failed'
        assert JobStatus.CANCELLED == 'cancelled'

    def test_all_contains_all_statuses(self):
        """ALL should contain all status values."""
        assert JobStatus.PENDING in JobStatus.ALL
        assert JobStatus.IN_PROGRESS in JobStatus.ALL
        assert JobStatus.COMPLETED in JobStatus.ALL
        assert JobStatus.FAILED in JobStatus.ALL
        assert JobStatus.CANCELLED in JobStatus.ALL
        assert len(JobStatus.ALL) == 5


class TestIsPreviewable:
    """Unit tests for is_previewable()"""

    def test_previewable_media_types(self):
        """Images, video, audio, 3d, text media types should be previewable."""
        for media_type in ['images', 'video', 'audio', '3d', 'text']:
            assert is_previewable(media_type, {}) is True

    def test_non_previewable_media_types(self):
        """Other media types should not be previewable."""
        for media_type in ['latents', 'metadata', 'files']:
            assert is_previewable(media_type, {}) is False

    def test_3d_extensions_previewable(self):
        """3D file extensions should be previewable regardless of media_type."""
        for ext in ['.obj', '.fbx', '.gltf', '.glb', '.usdz']:
            item = {'filename': f'model{ext}'}
            assert is_previewable('files', item) is True

    def test_3d_extensions_case_insensitive(self):
        """3D extension check should be case insensitive."""
        item = {'filename': 'MODEL.GLB'}
        assert is_previewable('files', item) is True

    def test_video_format_previewable(self):
        """Items with video/ format should be previewable."""
        item = {'format': 'video/mp4'}
        assert is_previewable('files', item) is True

    def test_audio_format_previewable(self):
        """Items with audio/ format should be previewable."""
        item = {'format': 'audio/wav'}
        assert is_previewable('files', item) is True

    def test_other_format_not_previewable(self):
        """Items with other format should not be previewable."""
        item = {'format': 'application/json'}
        assert is_previewable('files', item) is False


class TestGetOutputsSummary:
    """Unit tests for get_outputs_summary()"""

    def test_empty_outputs(self):
        """Empty outputs should return 0 count and None preview."""
        count, preview = get_outputs_summary({})
        assert count == 0
        assert preview is None

    def test_counts_across_multiple_nodes(self):
        """Outputs from multiple nodes should all be counted."""
        outputs = {
            'node1': {'images': [{'filename': 'a.png', 'type': 'output'}]},
            'node2': {'images': [{'filename': 'b.png', 'type': 'output'}]},
            'node3': {'images': [
                {'filename': 'c.png', 'type': 'output'},
                {'filename': 'd.png', 'type': 'output'}
            ]}
        }
        count, preview = get_outputs_summary(outputs)
        assert count == 4

    def test_skips_animated_key_and_non_list_values(self):
        """The 'animated' key and non-list values should be skipped."""
        outputs = {
            'node1': {
                'images': [{'filename': 'test.png', 'type': 'output'}],
                'animated': [True],  # Should skip due to key name
                'metadata': 'string',  # Should skip due to non-list
                'count': 42  # Should skip due to non-list
            }
        }
        count, preview = get_outputs_summary(outputs)
        assert count == 1

    def test_preview_prefers_type_output(self):
        """Items with type='output' should be preferred for preview."""
        outputs = {
            'node1': {
                'images': [
                    {'filename': 'temp.png', 'type': 'temp'},
                    {'filename': 'output.png', 'type': 'output'}
                ]
            }
        }
        count, preview = get_outputs_summary(outputs)
        assert count == 2
        assert preview['filename'] == 'output.png'

    def test_preview_fallback_when_no_output_type(self):
        """If no type='output', should use first previewable."""
        outputs = {
            'node1': {
                'images': [
                    {'filename': 'temp1.png', 'type': 'temp'},
                    {'filename': 'temp2.png', 'type': 'temp'}
                ]
            }
        }
        count, preview = get_outputs_summary(outputs)
        assert preview['filename'] == 'temp1.png'

    def test_non_previewable_media_types_counted_but_no_preview(self):
        """Non-previewable media types should be counted but not used as preview."""
        outputs = {
            'node1': {
                'latents': [
                    {'filename': 'latent1.safetensors'},
                    {'filename': 'latent2.safetensors'}
                ]
            }
        }
        count, preview = get_outputs_summary(outputs)
        assert count == 2
        assert preview is None

    def test_previewable_media_types(self):
        """Images, video, and audio media types should be previewable."""
        for media_type in ['images', 'video', 'audio']:
            outputs = {
                'node1': {
                    media_type: [{'filename': 'test.file', 'type': 'output'}]
                }
            }
            count, preview = get_outputs_summary(outputs)
            assert preview is not None, f"{media_type} should be previewable"

    def test_3d_files_previewable(self):
        """3D file extensions should be previewable."""
        for ext in ['.obj', '.fbx', '.gltf', '.glb', '.usdz']:
            outputs = {
                'node1': {
                    'files': [{'filename': f'model{ext}', 'type': 'output'}]
                }
            }
            count, preview = get_outputs_summary(outputs)
            assert preview is not None, f"3D file {ext} should be previewable"

    def test_format_mime_type_previewable(self):
        """Files with video/ or audio/ format should be previewable."""
        for fmt in ['video/x-custom', 'audio/x-custom']:
            outputs = {
                'node1': {
                    'files': [{'filename': 'file.custom', 'format': fmt, 'type': 'output'}]
                }
            }
            count, preview = get_outputs_summary(outputs)
            assert preview is not None, f"Format {fmt} should be previewable"

    def test_preview_enriched_with_node_metadata(self):
        """Preview should include nodeId, mediaType, and original fields."""
        outputs = {
            'node123': {
                'images': [{'filename': 'test.png', 'type': 'output', 'subfolder': 'outputs'}]
            }
        }
        count, preview = get_outputs_summary(outputs)
        assert preview['nodeId'] == 'node123'
        assert preview['mediaType'] == 'images'
        assert preview['subfolder'] == 'outputs'

    def test_string_3d_filename_creates_preview(self):
        """String items with 3D extensions should synthesize a preview (Preview3D node output).
        Only the .glb counts — nulls and non-file strings are excluded."""
        outputs = {
            'node1': {
                'result': ['preview3d_abc123.glb', None, None]
            }
        }
        count, preview = get_outputs_summary(outputs)
        assert count == 1
        assert preview is not None
        assert preview['filename'] == 'preview3d_abc123.glb'
        assert preview['mediaType'] == '3d'
        assert preview['nodeId'] == 'node1'
        assert preview['type'] == 'output'

    def test_string_non_3d_filename_no_preview(self):
        """String items without 3D extensions should not create a preview."""
        outputs = {
            'node1': {
                'result': ['data.json', None]
            }
        }
        count, preview = get_outputs_summary(outputs)
        assert count == 0
        assert preview is None

    def test_string_3d_filename_used_as_fallback(self):
        """String 3D preview should be used when no dict items are previewable."""
        outputs = {
            'node1': {
                'latents': [{'filename': 'latent.safetensors'}],
            },
            'node2': {
                'result': ['model.glb', None]
            }
        }
        count, preview = get_outputs_summary(outputs)
        assert preview is not None
        assert preview['filename'] == 'model.glb'
        assert preview['mediaType'] == '3d'


class TestHas3DExtension:
    """Unit tests for has_3d_extension()"""

    def test_recognized_extensions(self):
        for ext in ['.obj', '.fbx', '.gltf', '.glb', '.usdz']:
            assert has_3d_extension(f'model{ext}') is True

    def test_case_insensitive(self):
        assert has_3d_extension('MODEL.GLB') is True
        assert has_3d_extension('Scene.GLTF') is True

    def test_non_3d_extensions(self):
        for name in ['photo.png', 'video.mp4', 'data.json', 'model']:
            assert has_3d_extension(name) is False


class TestApplySorting:
    """Unit tests for apply_sorting()"""

    def test_sort_by_create_time_desc(self):
        """Default sort by create_time descending."""
        jobs = [
            {'id': 'a', 'create_time': 100},
            {'id': 'b', 'create_time': 300},
            {'id': 'c', 'create_time': 200},
        ]
        result = apply_sorting(jobs, 'created_at', 'desc')
        assert [j['id'] for j in result] == ['b', 'c', 'a']

    def test_sort_by_create_time_asc(self):
        """Sort by create_time ascending."""
        jobs = [
            {'id': 'a', 'create_time': 100},
            {'id': 'b', 'create_time': 300},
            {'id': 'c', 'create_time': 200},
        ]
        result = apply_sorting(jobs, 'created_at', 'asc')
        assert [j['id'] for j in result] == ['a', 'c', 'b']

    def test_sort_by_execution_duration(self):
        """Sort by execution_duration should order by duration."""
        jobs = [
            {'id': 'a', 'create_time': 100, 'execution_start_time': 100, 'execution_end_time': 5100},  # 5s
            {'id': 'b', 'create_time': 300, 'execution_start_time': 300, 'execution_end_time': 1300},  # 1s
            {'id': 'c', 'create_time': 200, 'execution_start_time': 200, 'execution_end_time': 3200},  # 3s
        ]
        result = apply_sorting(jobs, 'execution_duration', 'desc')
        assert [j['id'] for j in result] == ['a', 'c', 'b']

    def test_sort_with_none_values(self):
        """Jobs with None values should sort as 0."""
        jobs = [
            {'id': 'a', 'create_time': 100, 'execution_start_time': 100, 'execution_end_time': 5100},
            {'id': 'b', 'create_time': 300, 'execution_start_time': None, 'execution_end_time': None},
            {'id': 'c', 'create_time': 200, 'execution_start_time': 200, 'execution_end_time': 3200},
        ]
        result = apply_sorting(jobs, 'execution_duration', 'asc')
        assert result[0]['id'] == 'b'  # None treated as 0, comes first


class TestNormalizeQueueItem:
    """Unit tests for normalize_queue_item()"""

    def test_basic_normalization(self):
        """Queue item should be normalized to job dict."""
        item = (
            10,  # priority/number
            'prompt-123',  # prompt_id
            {'nodes': {}},  # prompt
            {
                'create_time': 1234567890,
                'extra_pnginfo': {'workflow': {'id': 'workflow-abc'}}
            },  # extra_data
            ['node1'],  # outputs_to_execute
        )
        job = normalize_queue_item(item, JobStatus.PENDING)

        assert job['id'] == 'prompt-123'
        assert job['status'] == 'pending'
        assert job['priority'] == 10
        assert job['create_time'] == 1234567890
        assert 'execution_start_time' not in job
        assert 'execution_end_time' not in job
        assert 'execution_error' not in job
        assert 'preview_output' not in job
        assert job['outputs_count'] == 0
        assert job['workflow_id'] == 'workflow-abc'


class TestNormalizeHistoryItem:
    """Unit tests for normalize_history_item()"""

    def test_completed_job(self):
        """Completed history item should have correct status and times from messages."""
        history_item = {
            'prompt': (
                5,  # priority
                'prompt-456',
                {'nodes': {}},
                {
                    'create_time': 1234567890000,
                    'extra_pnginfo': {'workflow': {'id': 'workflow-xyz'}}
                },
                ['node1'],
            ),
            'status': {
                'status_str': 'success',
                'completed': True,
                'messages': [
                    ('execution_start', {'prompt_id': 'prompt-456', 'timestamp': 1234567890500}),
                    ('execution_success', {'prompt_id': 'prompt-456', 'timestamp': 1234567893000}),
                ]
            },
            'outputs': {},
        }
        job = normalize_history_item('prompt-456', history_item)

        assert job['id'] == 'prompt-456'
        assert job['status'] == 'completed'
        assert job['priority'] == 5
        assert job['execution_start_time'] == 1234567890500
        assert job['execution_end_time'] == 1234567893000
        assert job['workflow_id'] == 'workflow-xyz'

    def test_failed_job(self):
        """Failed history item should have failed status and error from messages."""
        history_item = {
            'prompt': (
                5,
                'prompt-789',
                {'nodes': {}},
                {'create_time': 1234567890000},
                ['node1'],
            ),
            'status': {
                'status_str': 'error',
                'completed': False,
                'messages': [
                    ('execution_start', {'prompt_id': 'prompt-789', 'timestamp': 1234567890500}),
                    ('execution_error', {
                        'prompt_id': 'prompt-789',
                        'node_id': '5',
                        'node_type': 'KSampler',
                        'exception_message': 'CUDA out of memory',
                        'exception_type': 'RuntimeError',
                        'traceback': ['Traceback...', 'RuntimeError: CUDA out of memory'],
                        'timestamp': 1234567891000,
                    })
                ]
            },
            'outputs': {},
        }

        job = normalize_history_item('prompt-789', history_item)
        assert job['status'] == 'failed'
        assert job['execution_start_time'] == 1234567890500
        assert job['execution_end_time'] == 1234567891000
        assert job['execution_error']['node_id'] == '5'
        assert job['execution_error']['node_type'] == 'KSampler'
        assert job['execution_error']['exception_message'] == 'CUDA out of memory'

    def test_cancelled_job(self):
        """Cancelled/interrupted history item should have cancelled status."""
        history_item = {
            'prompt': (
                5,
                'prompt-cancelled',
                {'nodes': {}},
                {'create_time': 1234567890000},
                ['node1'],
            ),
            'status': {
                'status_str': 'error',
                'completed': False,
                'messages': [
                    ('execution_start', {'prompt_id': 'prompt-cancelled', 'timestamp': 1234567890500}),
                    ('execution_interrupted', {
                        'prompt_id': 'prompt-cancelled',
                        'node_id': '5',
                        'node_type': 'KSampler',
                        'executed': ['1', '2', '3'],
                        'timestamp': 1234567891000,
                    })
                ]
            },
            'outputs': {},
        }

        job = normalize_history_item('prompt-cancelled', history_item)
        assert job['status'] == 'cancelled'
        assert job['execution_start_time'] == 1234567890500
        assert job['execution_end_time'] == 1234567891000
        # Cancelled jobs should not have execution_error set
        assert 'execution_error' not in job

    def test_include_outputs(self):
        """When include_outputs=True, should include full output data."""
        history_item = {
            'prompt': (
                5,
                'prompt-123',
                {'nodes': {'1': {}}},
                {'create_time': 1234567890, 'client_id': 'abc'},
                ['node1'],
            ),
            'status': {'status_str': 'success', 'completed': True, 'messages': []},
            'outputs': {'node1': {'images': [{'filename': 'test.png'}]}},
        }
        job = normalize_history_item('prompt-123', history_item, include_outputs=True)

        assert 'outputs' in job
        assert 'workflow' in job
        assert 'execution_status' in job
        assert job['outputs'] == {'node1': {'images': [{'filename': 'test.png'}]}}
        assert job['workflow'] == {
            'prompt': {'nodes': {'1': {}}},
            'extra_data': {'create_time': 1234567890, 'client_id': 'abc'},
        }

    def test_include_outputs_normalizes_3d_strings(self):
        """Detail view should transform string 3D filenames into file output dicts."""
        history_item = {
            'prompt': (
                5,
                'prompt-3d',
                {'nodes': {}},
                {'create_time': 1234567890},
                ['node1'],
            ),
            'status': {'status_str': 'success', 'completed': True, 'messages': []},
            'outputs': {
                'node1': {
                    'result': ['preview3d_abc123.glb', None, None]
                }
            },
        }
        job = normalize_history_item('prompt-3d', history_item, include_outputs=True)

        assert job['outputs_count'] == 1
        result_items = job['outputs']['node1']['result']
        assert len(result_items) == 1
        assert result_items[0] == {
            'filename': 'preview3d_abc123.glb',
            'type': 'output',
            'subfolder': '',
            'mediaType': '3d',
        }

    def test_include_outputs_preserves_dict_items(self):
        """Detail view normalization should pass dict items through unchanged."""
        history_item = {
            'prompt': (
                5,
                'prompt-img',
                {'nodes': {}},
                {'create_time': 1234567890},
                ['node1'],
            ),
            'status': {'status_str': 'success', 'completed': True, 'messages': []},
            'outputs': {
                'node1': {
                    'images': [
                        {'filename': 'photo.png', 'type': 'output', 'subfolder': ''},
                    ]
                }
            },
        }
        job = normalize_history_item('prompt-img', history_item, include_outputs=True)

        assert job['outputs_count'] == 1
        assert job['outputs']['node1']['images'] == [
            {'filename': 'photo.png', 'type': 'output', 'subfolder': ''},
        ]


class TestNormalizeOutputItem:
    """Unit tests for normalize_output_item()"""

    def test_none_returns_none(self):
        assert normalize_output_item(None) is None

    def test_string_3d_extension_synthesizes_dict(self):
        result = normalize_output_item('model.glb')
        assert result == {'filename': 'model.glb', 'type': 'output', 'subfolder': '', 'mediaType': '3d'}

    def test_string_non_3d_extension_returns_none(self):
        assert normalize_output_item('data.json') is None

    def test_string_no_extension_returns_none(self):
        assert normalize_output_item('camera_info_string') is None

    def test_dict_passes_through(self):
        item = {'filename': 'test.png', 'type': 'output'}
        assert normalize_output_item(item) is item

    def test_other_types_return_none(self):
        assert normalize_output_item(42) is None
        assert normalize_output_item(True) is None


class TestNormalizeOutputs:
    """Unit tests for normalize_outputs()"""

    def test_empty_outputs(self):
        assert normalize_outputs({}) == {}

    def test_dict_items_pass_through(self):
        outputs = {
            'node1': {
                'images': [{'filename': 'a.png', 'type': 'output'}],
            }
        }
        result = normalize_outputs(outputs)
        assert result == outputs

    def test_3d_string_synthesized(self):
        outputs = {
            'node1': {
                'result': ['model.glb', None, None],
            }
        }
        result = normalize_outputs(outputs)
        assert result == {
            'node1': {
                'result': [
                    {'filename': 'model.glb', 'type': 'output', 'subfolder': '', 'mediaType': '3d'},
                ],
            }
        }

    def test_animated_key_preserved(self):
        outputs = {
            'node1': {
                'images': [{'filename': 'a.png', 'type': 'output'}],
                'animated': [True],
            }
        }
        result = normalize_outputs(outputs)
        assert result['node1']['animated'] == [True]

    def test_non_dict_node_outputs_preserved(self):
        outputs = {'node1': 'unexpected_value'}
        result = normalize_outputs(outputs)
        assert result == {'node1': 'unexpected_value'}

    def test_none_items_filtered_but_other_types_preserved(self):
        outputs = {
            'node1': {
                'result': ['data.json', None, [1, 2, 3]],
            }
        }
        result = normalize_outputs(outputs)
        assert result == {
            'node1': {
                'result': ['data.json', [1, 2, 3]],
            }
        }
