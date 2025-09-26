import random

def generate_quad_block():
    block = bytearray(128)  # 1024 bits = 128 bytes
    for slot in range(4):
        length = 7  # Fixed length 7 bytes
        msg = bytes([random.randint(0, 255) for _ in range(length)])
        offset = (3 - slot) * 32  # Adjusted to match C++ range: MSB first, so reverse slot order
        block[offset] = length
        block[offset + 1 : offset + 1 + length] = msg
    return block

def write_input_file(filename, num_blocks):
    with open(filename, "wb") as f:
        for i in range(num_blocks):
            f.write(generate_quad_block())
            if i % 100000 == 0 and i > 0:
                print(f"Written {i} blocks...")

if __name__ == "__main__":
    n_msgs = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216]
    for n_msg in n_msgs:
        num_blocks = n_msg // 4
        filename = f"input_{n_msg}.dat"
        print(f"Generating {filename} with {num_blocks} blocks ({n_msg} messages)")
        write_input_file(filename, num_blocks)
        print(f"Finished {filename}")