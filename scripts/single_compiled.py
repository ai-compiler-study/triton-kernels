import torch

from triton_kernels.flux import SingleStreamBlock

hidden_size = 3072
num_heads = 24
mlp_ratio = 4.0
head_dim = hidden_size // num_heads

batch_size = 3
seq_len = 4336

device = "cuda"

block = SingleStreamBlock(
    hidden_size=hidden_size,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
)
block_compiled = SingleStreamBlock(
    hidden_size=hidden_size,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
)
block_compiled.load_state_dict(block.state_dict())
block_compiled = torch.compile(block_compiled)
block = block.to(device)
block_compiled = block_compiled.to(device)

x = torch.randn([batch_size, seq_len, hidden_size], device=device)
vec = torch.randn([batch_size, hidden_size], device=device)
pe = torch.randn([1, 1, seq_len, head_dim // 2, 2, 2], device=device)

out = block(x=x, vec=vec, pe=pe)
out_compiled = block_compiled(x=x, vec=vec, pe=pe)

torch.testing.assert_close(out, out_compiled, atol=1e-5, rtol=0)

# warmup
warmup_count = 5

for i in range(warmup_count):
    out = block(x=x, vec=vec, pe=pe)
for i in range(warmup_count):
    out_compiled = block_compiled(x=x, vec=vec, pe=pe)

# run
run_count = 100

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for i in range(run_count):
    out = block(x=x, vec=vec, pe=pe)
end.record()
torch.cuda.synchronize()
print(f"baseline block time: {start.elapsed_time(end):.2f} ms")

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for i in range(run_count):
    out_compiled = block_compiled(x=x, vec=vec, pe=pe)
end.record()
torch.cuda.synchronize()
print(f"compiled block time: {start.elapsed_time(end):.2f} ms")
