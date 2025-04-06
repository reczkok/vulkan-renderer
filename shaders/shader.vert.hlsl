struct VSInput {
    uint vertexID : SV_VertexID;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 color : COLOR0;
};

static const float2 positions[3] = {
    float2(0.0f, -0.5f),
    float2(0.5f, 0.5f),
    float2(-0.5f, 0.5f)
};

static const float3 colors[3] = {
    float3(1.0f, 0.0f, 0.0f),
    float3(0.0f, 1.0f, 0.0f),
    float3(0.0f, 0.0f, 1.0f)
};

VSOutput main(VSInput input) {
    VSOutput output;
    output.position = float4(positions[input.vertexID], 0.0f, 1.0f);
    output.color = colors[input.vertexID];
    return output;
}