#include <cstdint>
#include <cmath>
#include <iostream>
#include <bitset>

class half {
public:
    uint16_t value;

    half() : value(0) {}

    half(uint16_t v) : value(v) {}

    half(float f) {
        // 简单的浮点数转换到半精度
        // 实际实现需要处理溢出、舍入等情况
        int s = (f < 0) ? 1 : 0;
        f = fabs(f);
        int exp = 0;
        
        if (f >= 1.0f) {
            while (f >= 2.0f) {
                f /= 2.0f;
                exp++;
            }
        } else if (f < 1.0f && f > 0.0f) {
            while (f < 0.5f) {
                f *= 2.0f;
                exp--;
            }
        }
        
        int mantissa = static_cast<int>(f * (1 << 10));  // 10 位有效数字
        
        value = (s << 15) | ((exp + 15) << 10) | (mantissa & 0x3FF);
    }

    float to_float() const {
        // 将 half 转换为 float
        int s = (value >> 15) & 0x1;
        int exp = (value >> 10) & 0x1F;
        int mantissa = value & 0x3FF;

        if (exp == 0) return 0.0f;  // 处理零情况
        float f = static_cast<float>(mantissa) / (1 << 10);
        f += 1.0f;  // 加上隐含的 1
        f *= pow(2.0f, exp - 15);  // 处理指数

        return s ? -f : f;  // 处理符号
    }

    void print_binary() const {
        std::bitset<16> b(value);
        std::cout << "Binary representation: " << b << std::endl;
    }

    void print_hex() const {
        std::cout << "Hexadecimal representation: 0x" << std::hex << value << std::endl;
    }

    half operator-(const half& other) const {
        return half(this->to_float() - other.to_float());
    }
};

int main() {
    
    // TEST PRINT
    // half h1 = 1536.0f;
    // half h2 = -2.5f;

    // std::cout << "h1 as float: " << h1.to_float() << std::endl;
    // h1.print_binary();
    // h1.print_hex();

    // std::cout << "h2 as float: " << h2.to_float() << std::endl;
    // h2.print_binary();
    // h1.print_hex();


    /** P1
     * int8_t input, half output
     * int8_t bias = 0x6600 
     * half magic = 1536   (half类型的 1536 对应的是十六进制为 0x6600)
     * 
     * 通过  output = (half)(input + bias) - magic可以实现 int8_t => half 的数值转化
     */

    // ---- TEST P1 ----
    int8_t num_int8 = 120;

    half magic = 1536.0f;
    int16_t bais = 0x6600;

    // std::cout << "Hexadecimal representation: 0x" << std::hex << (num_int8 + bais) << std::endl;

    half num_half = half(static_cast<uint16_t>(num_int8 + bais)) - magic;
    std::cout << num_half.to_float() << std::endl;


    /** P2
     * int8_t bias1 = 0x5400
     * half magic1 = 64 (half类型的 64 对应的是十六进制为 0x5400)
     *
     * 对于插入 int8 变量的 int4 高位部分: 0xX0
     * 通过  (half)(0xX0 + 0x5400) - (half)(0x5400) = (half) X
     * 可以取出 half 值 X
     */

    // ---- TEST P2 ----
    int8_t num_int4_high = 0x70;

    half magic1 = 64.0f;
    int16_t bais1 = 0x5400;

    // std::cout << "Hexadecimal representation: 0x" << std::hex << (num_int4_high + bais1) << std::endl;

    half num_half_high = half(static_cast<uint16_t>(num_int4_high + bais1)) - magic1;
    std::cout << num_half_high.to_float() << std::endl;

    // Conclusion
    /** int8 对称量化: w_half = (w_int8 - zero) * scale
     *  int8_t bias = 0x6600
        half magic = 1536
 
        w_half = ((half)(w_int8 - zero + bias) - magic) * scale                               --- 公式1
               = (half)(w_int8 - zero + bias) * scale - magic * scale

               = (((half)(w_int8 + bias) - magic) -  ((half)(zero + bias) - magic)) * scale   --- 公式2
               = (half)(w_int8 + bias) * scale - (half)(zero + bias) * scale
        
        其中 公式 2 可以适用于 非对称量化, 能够避免 w_int8 - zero 可能出现的溢出问题.
     */

    /** int4 非对称/对称量化
     * int8_t bias0 = 0x6600
       half magic0 = 1536
 
       w_low_half = (((half)(w_int8_000F + bias0) - magic0) -  ((half)(zero + bias0) - magic0)) * scale   --- 采用公式2
              = (half)(w_int8_000F + bias0) * scale - (half)(zero + bias0) * scale

       int8_t bias1 = 0x5400
       half magic1 = 64

       w_high_half = (((half)(w_int8_00F0 + bias1) - magic1) -  ((half)(zero * 128 + bias1) - magic1)) * scale   --- 采用公式2
              = (half)(w_int8_00F0 + bias1) * scale - (half)(zero * 128 + bias1) * scale

       其中存在对应关系:
       (half)(zero * 128 + bias1) - (half)(zero + bias0) = (half)1472
     */

    return 0;
}