Shader "Custom/TriangleSoupRasterizer"
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

			// Uniforms
			int _CurrentFrame;
			int _PrimitiveCount;
			float _SimpleColorRender;
			float4x4 _CameraMatrixVP;
			float _DebugTriangleView;


			// ========================== VERTEX SHADER ==========================
			struct PrimitiveData
			{
				float3 positions[3];
				float3 color;
			};

			StructuredBuffer<PrimitiveData> _PrimitiveBuffer;

			struct Varyings
			{
				float4 position : SV_POSITION;
				float3 color : COLOR;
				uint primitiveID : PRIMITIVEID;
			};

			Varyings Vertex(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
			{
				// Retrieve vertex attributes
				PrimitiveData primitiveData = _PrimitiveBuffer[instanceID];
				float3 worldPos = primitiveData.positions[vertexID];

				// Camera projection
				float4 clipPos = mul(_CameraMatrixVP, float4(worldPos, 1));

				// Output
				Varyings output;
				output.position = clipPos;
				output.color = primitiveData.color;
				output.primitiveID = instanceID;
				return output;
			}



			// ========================== FRAGMENT SHADER ==========================
			struct FragmentOutput
			{
				float4 color : SV_Target0;
				float id : SV_Target1;
			};

			FragmentOutput Fragment(Varyings input) : SV_Target
			{
				FragmentOutput output;
				output.color = float4(input.color, 1);
				output.color.rgb = pow(output.color.rgb, 2.2);
				output.id = asfloat(input.primitiveID);
				return output;
			}
			ENDCG
		}
	}
	CustomEditor "Pcx.DiskMaterialInspector"
}
