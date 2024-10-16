import torch
import triton
import triton.language as tl

@triton.jit
def triton_qkv_concat(txt_qkv, img_qkv, out_q_ptr, out_k_ptr, out_v_ptr,
                      seq_len, num_heads, head_dim, hidden_dim, seq_txt_len,
                      stride_txt_a, stride_txt_b,
                      stride_img_a, stride_img_b,
                      stride_output_a, stride_output_b, stride_output_c,
                      XBLOCK : tl.constexpr):
    
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK + tl.arange(0, XBLOCK)[:]

    seq_idx = (xoffset // hidden_dim) % seq_len
    batch_idx = (xoffset // stride_output_a)
    hidden_dim_idx = xoffset % hidden_dim
    headdim_idx = xoffset % head_dim
    head_idx = (xoffset // head_dim) % num_heads

    txt_seq_end = tl.full([1], seq_txt_len, tl.int64)
    txt_mask = seq_idx < txt_seq_end
    img_mask = seq_idx >= txt_seq_end

    # compute query
    txt_q_data = tl.load(txt_qkv + (hidden_dim*0 + hidden_dim_idx + (stride_txt_b*seq_idx) + (stride_txt_a*batch_idx)), txt_mask, other=0.0).to(tl.float32)
    zero_mask = tl.full(txt_q_data.shape, 0.0, txt_q_data.dtype)
    masked_txt_q = tl.where(txt_mask, txt_q_data, zero_mask) 

    img_q_data = tl.load(img_qkv + (((-stride_txt_a + hidden_dim * 0)) + hidden_dim_idx + (stride_img_b*seq_idx) + (stride_img_a*batch_idx)), img_mask, other=0.0).to(tl.float32)
    zero_mask = tl.full(img_q_data.shape, 0.0, img_q_data.dtype)
    masked_img_q = tl.where(img_mask, img_q_data, zero_mask)

    out_q = tl.where(txt_mask, masked_txt_q, masked_img_q)
    tl.store(out_q_ptr + (headdim_idx + (stride_output_c*seq_idx) + (stride_output_b*head_idx) + (stride_output_a*batch_idx)), out_q, None)

    # compute key
    txt_k_data = tl.load(txt_qkv + (hidden_dim*1 + hidden_dim_idx + (stride_txt_b*seq_idx) + (stride_txt_a*batch_idx)), txt_mask, other=0.0).to(tl.float32)
    zero_mask = tl.full(txt_k_data.shape, 0.0, txt_k_data.dtype)
    masked_txt_q = tl.where(txt_mask, txt_k_data, zero_mask)
   
    img_k_data = tl.load(img_qkv + (((-stride_txt_a + hidden_dim * 1)) + hidden_dim_idx + (stride_img_b*seq_idx) + (stride_img_a*batch_idx)), img_mask, other=0.0).to(tl.float32)
    zero_mask = tl.full(img_k_data.shape, 0.0, img_k_data.dtype)
    masked_img_k = tl.where(img_mask, img_k_data, zero_mask)
 
    out_k = tl.where(txt_mask, masked_txt_q, masked_img_k)
    tl.store(out_k_ptr + (headdim_idx + (stride_output_c*seq_idx) + (stride_output_b*head_idx) + (stride_output_a*batch_idx)), out_k, None)

    # compute value
    txt_v_data = tl.load(txt_qkv + (hidden_dim*2 + hidden_dim_idx + (stride_txt_b*seq_idx) + (stride_txt_a*batch_idx)), txt_mask, other=0.0).to(tl.float32)
    zero_mask = tl.full(txt_v_data.shape, 0.0, txt_v_data.dtype)
    masked_txt_v = tl.where(txt_mask, txt_v_data, zero_mask)
   
    img_v_data = tl.load(img_qkv + (((-stride_txt_a + hidden_dim * 2)) + hidden_dim_idx + (stride_img_b*seq_idx) + (stride_img_a*batch_idx)), img_mask, other=0.0).to(tl.float32)
    zero_mask = tl.full(img_v_data.shape, 0.0, img_v_data.dtype)
    masked_img_q = tl.where(img_mask, img_v_data, zero_mask)
    
    output_v = tl.where(txt_mask, masked_txt_v, masked_img_q)
    tl.store(out_v_ptr + (headdim_idx + (stride_output_c*seq_idx) + (stride_output_b*head_idx) + (stride_output_a*batch_idx)), output_v, None)