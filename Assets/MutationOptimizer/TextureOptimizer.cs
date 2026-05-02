using UnityEngine;
using Unity.Mathematics;

public class TextureOptimizer
{
    // Shaders
    public Material rasterMaterial;
    public ComputeShader textureOptimizerCS;
    
    // Compute kernels
    public int kernelReset;
    public int kernelRandomPerturbation;
    public int kernelGradientEstimation;
    public int kernelGradientEstimationPost;
    public int kernelGradientDescent;
    
    // Resources
    public RenderTexture albedoMap;
    public RenderTexture albedoMapMutated;
    public RenderTexture uvBufferMutated;
    public ComputeBuffer primitiveBufferMutated;
    private ComputeBuffer optimStepGradientsBuffer;
    private ComputeBuffer optimStepMutationError;
    private ComputeBuffer gradientMoments1Buffer;
    private ComputeBuffer gradientMoments2Buffer;
    private ComputeBuffer optimStepCounterBuffer;

    public int texelCounts;
    public Vector2Int targetResolution;
    
    private Color UVClearColor = new Color(4294967295, 0, 0, 0);
    
    public TextureOptimizer(Vector2Int targetResolution)
    {
        this.targetResolution = targetResolution;
        
        // Init rendering material
        rasterMaterial = new Material(Shader.Find("Custom/TexturedMeshRasterizer"));
        rasterMaterial.hideFlags = HideFlags.DontSave;
        
        textureOptimizerCS = Resources.Load<ComputeShader>("TextureOptimizer");
        
        kernelReset = textureOptimizerCS.FindKernel("Reset");
        kernelRandomPerturbation = textureOptimizerCS.FindKernel("RandomPerturbation");
        kernelGradientEstimation = textureOptimizerCS.FindKernel("GradientEstimation");
        kernelGradientEstimationPost = textureOptimizerCS.FindKernel("GradientEstimationPost");
        kernelGradientDescent = textureOptimizerCS.FindKernel("GradientDescent");
        
        InitResources();
    }
    
    public void ResetOptimizationStep()
    {
        textureOptimizerCS.SetBuffer(kernelReset, "_PrimitiveGradientsOptimStep", optimStepGradientsBuffer);
        textureOptimizerCS.SetBuffer(kernelReset, "_PrimitiveMutationError", optimStepMutationError);
        StochasticOptimizer.DispatchCompute1D(textureOptimizerCS, kernelReset, texelCounts, 256);
    }
    
    public void DoRandomPerturbation()
    {
        textureOptimizerCS.SetTexture(kernelRandomPerturbation, "_AlbedoMap", albedoMap);
        textureOptimizerCS.SetTexture(kernelRandomPerturbation, "_AlbedoMapMutated",  albedoMapMutated);
        StochasticOptimizer.DispatchCompute1D(textureOptimizerCS, kernelRandomPerturbation, texelCounts, 256);
    } 
    
    public void DoGradientEstimation(RenderTexture targetFrameBuffer, RenderTexture renderedFrameMutatedMinus, RenderTexture renderedFrameMutatedPlus) 
    { 
        // Accumulate per pixel image loss
        textureOptimizerCS.SetTexture(kernelGradientEstimation, "_AlbedoMap", albedoMap);
        textureOptimizerCS.SetTexture(kernelGradientEstimation, "_TargetTexture", targetFrameBuffer);
        textureOptimizerCS.SetTexture(kernelGradientEstimation, "_ResolvedFrameMutatedMinus", renderedFrameMutatedMinus);
        textureOptimizerCS.SetTexture(kernelGradientEstimation, "_ResolvedFrameMutatedPlus", renderedFrameMutatedPlus);
        textureOptimizerCS.SetBuffer(kernelGradientEstimation, "_PrimitiveMutationError", optimStepMutationError);
        textureOptimizerCS.SetTexture(kernelGradientEstimation, "_UVBufferMutated", uvBufferMutated);
        textureOptimizerCS.Dispatch(kernelGradientEstimation, (int)math.ceil(targetResolution.x / 16.0f), (int)math.ceil(targetResolution.y / 16.0f), 1);

        // Accumulate gradients
        textureOptimizerCS.SetTexture(kernelGradientEstimationPost, "_AlbedoMap", albedoMap);
        textureOptimizerCS.SetTexture(kernelGradientEstimationPost, "_AlbedoMapMutated", albedoMapMutated);
        textureOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_PrimitiveMutationError", optimStepMutationError);
        textureOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_PrimitiveGradientsOptimStep", optimStepGradientsBuffer); 
        StochasticOptimizer.DispatchCompute1D(textureOptimizerCS, kernelGradientEstimationPost, texelCounts, 256); 
    }
    
    public void DoGradientDescent()
    {
        textureOptimizerCS.SetTexture(kernelGradientDescent, "_AlbedoMap", albedoMap);
        textureOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveGradientsMoments1", gradientMoments1Buffer);
        textureOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveGradientsMoments2", gradientMoments2Buffer);
        textureOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveGradientsOptimStep", optimStepGradientsBuffer);
        textureOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveOptimStepCounter", optimStepCounterBuffer);
        textureOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveMutationError", optimStepMutationError);
        StochasticOptimizer.DispatchCompute1D(textureOptimizerCS, kernelGradientDescent, texelCounts, 256);
    }
    
    public void RenderOptimizationSceneWithUVs(Camera cameraToUse, RenderTexture renderTargetToUse, RenderTexture uvRenderTargetToUse, Mesh m)
    {
        rasterMaterial.SetTexture("_MainTex", albedoMapMutated);
        
        // Camera setup
        Matrix4x4 cameraVP = GL.GetGPUProjectionMatrix(cameraToUse.projectionMatrix, true) * cameraToUse.worldToCameraMatrix;
        rasterMaterial.SetMatrix("_CameraMatrixVP", cameraVP);

        // Clear targets
        Graphics.SetRenderTarget(uvRenderTargetToUse.colorBuffer, uvRenderTargetToUse.depthBuffer);
        GL.Clear(true, true, UVClearColor, 1.0f);
        Graphics.SetRenderTarget(renderTargetToUse.colorBuffer, renderTargetToUse.depthBuffer);
        GL.Clear(true, true, Color.clear, 1.0f);

        // Render both color + UV
        RenderBuffer[] mrt = new RenderBuffer[] { renderTargetToUse.colorBuffer, uvRenderTargetToUse.colorBuffer };
        Graphics.SetRenderTarget(mrt, uvRenderTargetToUse.depthBuffer);
        rasterMaterial.SetPass(0);
        Graphics.DrawMeshNow(m, Matrix4x4.identity);
    }

    public void InitResources()
    {
        ReleaseResources();
        
        // Init albedoMap
        Texture2D white = Resources.Load<Texture2D>("1024");
        albedoMap = new RenderTexture(white.width, white.height, 0, RenderTextureFormat.ARGB32);
        albedoMap.enableRandomWrite = true;
        albedoMap.autoGenerateMips = true;
        albedoMap.filterMode = FilterMode.Point;
        albedoMap.Create();
        Graphics.Blit(white, albedoMap);
        texelCounts = albedoMap.width * albedoMap.height;
        
        albedoMapMutated = new RenderTexture(white.width, white.height, 0, RenderTextureFormat.ARGB32);
        albedoMapMutated.enableRandomWrite = true;
        albedoMapMutated.autoGenerateMips = true;
        albedoMapMutated.filterMode = FilterMode.Point;
        albedoMapMutated.Create();
        
        // Init everything
        uvBufferMutated = new RenderTexture(targetResolution.x, targetResolution.y, 32, RenderTextureFormat.RGFloat, RenderTextureReadWrite.Linear);
        uvBufferMutated.enableRandomWrite = true;
        
        // Optim Buffers
        int byteSize = 12; // int3 or float3
        optimStepGradientsBuffer = new ComputeBuffer(texelCounts, byteSize);
        gradientMoments1Buffer = new ComputeBuffer(texelCounts, byteSize);
        gradientMoments2Buffer = new ComputeBuffer(texelCounts, byteSize);
        primitiveBufferMutated = new ComputeBuffer(texelCounts, byteSize);
        optimStepMutationError = new ComputeBuffer(texelCounts, sizeof(int));
        optimStepCounterBuffer = new ComputeBuffer(texelCounts, sizeof(int));
        StochasticOptimizer.ZeroInitBuffer(optimStepGradientsBuffer);
        StochasticOptimizer.ZeroInitBuffer(gradientMoments1Buffer);
        StochasticOptimizer.ZeroInitBuffer(gradientMoments2Buffer);
        StochasticOptimizer.ZeroInitBuffer(primitiveBufferMutated);
        StochasticOptimizer.ZeroInitBuffer(optimStepMutationError);
        StochasticOptimizer.ZeroInitBuffer(optimStepCounterBuffer);
    }

    public void ReleaseResources()
    {
        StochasticOptimizer.SafeRelease(ref uvBufferMutated);
        StochasticOptimizer.SafeRelease(ref primitiveBufferMutated);
        StochasticOptimizer.SafeRelease(ref optimStepGradientsBuffer);
        StochasticOptimizer.SafeRelease(ref optimStepMutationError);
        StochasticOptimizer.SafeRelease(ref gradientMoments1Buffer);
        StochasticOptimizer.SafeRelease(ref gradientMoments2Buffer);
        StochasticOptimizer.SafeRelease(ref optimStepCounterBuffer);
    }

    public void ReleaseAllResources()
    {
        ReleaseResources();
        StochasticOptimizer.SafeRelease(ref albedoMap);
        StochasticOptimizer.SafeRelease(ref albedoMapMutated);
        StochasticOptimizer.SafeDestroy(ref rasterMaterial);
    }

}
