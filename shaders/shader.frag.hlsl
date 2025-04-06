struct PSInput {
    [[vk::location(0)]] float3 fragColor : COLOR0;
};

struct PSOutput {
    [[vk::location(0)]] float4 outColor : SV_Target;
};

PSOutput main(PSInput input) {
    PSOutput output;
    output.outColor = float4(input.fragColor, 1.0f);
    return output;
}