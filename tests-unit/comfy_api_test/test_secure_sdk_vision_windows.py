import asyncio
from types import SimpleNamespace
import pytest
import torch
from comfy_api.latest import _sdk

async def window(values,field,start,count):
    refs=_sdk.InProcessRefResolver()
    output=_sdk.ClipVisionOutputRef._wrap(await refs.create('CLIP_VISION_OUTPUT',SimpleNamespace(**{field:values})))
    with _sdk.bind_runtime(refs,None,_sdk.InProcessOps()):
        result=await getattr(output,field)(batch_start=start,batch_count=count)
        return await refs.resolve(result)

@pytest.mark.parametrize('field',['last_hidden_state','penultimate_hidden_states'])
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64,torch.bfloat16])
def test_windows_reassemble_exact_tokens_and_own_their_storage(field,dtype):
    values=torch.arange(256*9*5,dtype=dtype).reshape(256,9,5).transpose(1,2)
    slices=[asyncio.run(window(values,field,start,64)) for start in range(0,256,64)]
    assert torch.equal(torch.cat(slices),values)
    for part in slices:
        assert part.dtype==dtype and part.device==values.device
        assert part.untyped_storage().nbytes()==part.numel()*part.element_size()
        assert part.untyped_storage().data_ptr()!=values.untyped_storage().data_ptr()
    before=values.clone();slices[0].fill_(0);assert torch.equal(values,before)

@pytest.mark.parametrize('start,count',[(-1,1),(256,1),(250,7),(0,65),(0,0),(True,1),(0,True),('0',1),(0,1.0),(1,None)])
def test_invalid_or_unbounded_windows_fail_before_copy(start,count,monkeypatch):
    values=torch.empty((256,7,3),device='meta')
    monkeypatch.setattr(torch.Tensor,'clone',lambda *a,**kw:pytest.fail('copy attempted'))
    with pytest.raises(ValueError,match='bounded'):asyncio.run(window(values,'last_hidden_state',start,count))

def test_selected_window_size_keeps_per_buffer_ceiling():
    values=torch.empty((256,577,1024),dtype=torch.float32,device='meta')
    selected=asyncio.run(window(values,'last_hidden_state',192,64))
    assert selected.shape==(64,577,1024)
    assert selected.numel()*selected.element_size()<512*1024*1024
    oversized=torch.empty((256,2049,1024),device='meta')
    with pytest.raises(ValueError,match='bounded'):asyncio.run(window(oversized,'last_hidden_state',0,64))

def test_missing_count_uses_remaining_bounded_rows():
    values=torch.arange(256*2*3).float().reshape(256,2,3)
    result=asyncio.run(window(values,'last_hidden_state',240,None))
    assert torch.equal(result,values[240:])
