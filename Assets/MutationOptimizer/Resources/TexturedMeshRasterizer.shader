Shader "Custom/TexturedMeshRasterizer"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD1;
            };

            struct v2f
            {
                float2 uv : TEXCOORD1;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4x4 _CameraMatrixVP;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = mul(_CameraMatrixVP, v.vertex);
                o.uv = v.uv;
                return o;
            }

            struct fragOutput
            {
                float4 color : SV_Target0;
                float2 uv : SV_Target1;
            };

            fragOutput frag (v2f i) : SV_Target
            {
                fragOutput o;

                o.color = tex2D(_MainTex, i.uv);
                o.uv = i.uv;

                return o;
            }
            ENDCG
        }
    }
}
