#pragma once

#define EQ(x, y) ((((0U - ((unsigned)(x) ^ (unsigned)(y))) >> 8) & 0xFF) ^ 0xFF)
#define GT(x, y) ((((unsigned)(y) - (unsigned)(x)) >> 8) & 0xFF)
#define GE(x, y) (GT(y, x) ^ 0xFF)
#define LT(x, y) GT(y, x)
#define LE(x, y) GE(y, x)

inline int b64_byte_to_char(unsigned x) {
    return (LT(x, 26) & (x + 'A')) |
        (GE(x, 26) & LT(x, 52) & (x + ('a' - 26))) |
        (GE(x, 52) & LT(x, 62) & (x + ('0' - 52))) | (EQ(x, 62) & '+') |
        (EQ(x, 63) & '/');
}

inline void to_base64_(char *dst, size_t dst_len, const void *src, size_t src_len) {
    size_t olen;
    const unsigned char *buf;
    unsigned acc, acc_len;

    olen = (src_len / 3) << 2;
    switch (src_len % 3) {
    case 2:
        olen++;
        /* fall through */
    case 1:
        olen += 2;
        break;
    }
    if (dst_len <= olen) {
        throw std::logic_error("SHORTTTTTTTTTTTTTTTTT");
        return;
    }
    acc = 0;
    acc_len = 0;
    buf = (const unsigned char *)src;
    while (src_len-- > 0) {
        acc = (acc << 8) + (*buf++);
        acc_len += 8;
        while (acc_len >= 6) {
            acc_len -= 6;
            *dst++ = (char)b64_byte_to_char((acc >> acc_len) & 0x3F);
        }
    }
    if (acc_len > 0) {
        *dst++ = (char)b64_byte_to_char((acc << (6 - acc_len)) & 0x3F);
    }
    *dst++ = 0;
}