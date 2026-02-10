Shader "Custom/MeshRasterizer"
{
	Properties
	{

	}
	SubShader
	{
		Tags { "RenderType" = "Opaque" }
		ZWrite On
		ZTest Less
		Cull Off
		//Conservative True

		Pass
		{
			CGPROGRAM
			#pragma target 5.0
			#pragma vertex Vertex
			#pragma fragment Fragment
			#include "UnityCG.cginc"
			#pragma enable_d3d11_debug_symbols

			// Uniforms
			int _VertexCount;
			float4x4 _CameraMatrixVP;
			
			int _EnableWireframe;
			float _WireWidth;


			// ========================== VERTEX SHADER ==========================
			StructuredBuffer<float3> _PrimitiveBuffer;
			StructuredBuffer<int> _IndexBuffer;

			struct Varyings
			{
				float4 position : SV_POSITION;
				float3 color : COLOR;
				float3 bary : TEXCOORD0;
				uint primitiveID : PRIMITIVEID;
			};

			Varyings Vertex(uint vertexID : SV_VertexID)
			{
				// Retrieve vertex attributes
				int posID = _IndexBuffer[vertexID];
				float3 worldPos =  _PrimitiveBuffer[posID];

				// Camera projection
				float4 clipPos = mul(_CameraMatrixVP, float4(worldPos, 1));

				// Barycentric coordinates from vertexID
				uint vid = vertexID % 3;
				float3 bary = (vid == 0) ? float3(1, 0, 0) :
							  (vid == 1) ? float3(0, 1, 0) :
								           float3(0, 0, 1);

				// Output
				Varyings o;
				o.position = clipPos;
				o.color = _PrimitiveBuffer[_VertexCount + vertexID / 3];
				o.bary = bary;
				o.primitiveID = vertexID / 3;
				return o;
			}



			// ========================== FRAGMENT SHADER ==========================
			struct FragmentOutput
			{
				float4 color : SV_Target0;
				float id : SV_Target1;
			};

			FragmentOutput Fragment(Varyings input) : SV_Target
			{
                FragmentOutput o;

                // base color
                float3 baseColor = pow(input.color, 2.2);

                // wireframe factor (1 = face, 0 = edge)
                float edge = min(min(input.bary.x, input.bary.y), input.bary.z);
                float w = fwidth(edge) * _WireWidth;
                float wireFactor = smoothstep(0.0, w, edge);

                // enable switch
                wireFactor = lerp(1.0, wireFactor, (float)_EnableWireframe);

                o.color = float4(baseColor * wireFactor, 1.0);
                o.id = asfloat(input.primitiveID);
                return o;
			}
			ENDCG
		}
	}
	CustomEditor "Pcx.DiskMaterialInspector"
}
