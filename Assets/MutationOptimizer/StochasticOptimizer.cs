using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
#if UNITY_EDITOR
using Unity.Burst;
#endif
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;


public class StochasticOptimizer : MonoBehaviour
{
	// ======================= PRIVATE VARIABLES =======================
	public RenderTexture idBufferMutatedMinus;
	public RenderTexture idBufferMutatedPlus;
	public RenderTexture renderedFrameMutatedMinus;
	public RenderTexture renderedFrameMutatedPlus;
	public RenderTexture renderedFrameFreeView;
	public RenderTexture targetFrameBuffer;
	private Material rasterMaterial;
	private Camera cameraDisplay;
	private Camera cameraOptim;
	private ComputeShader stochasticOptimizerCS;
	private ComputeShader triangleResamplingCS;
	private ComputeBuffer indexBuffer;
	private ComputeBuffer AdjacencyBuffer;
	private ComputeBuffer AdjacencyStartBuffer;
	private ComputeBuffer AdjacencyCountBuffer;
	private ComputeBuffer primitiveBuffer;
	private ComputeBuffer primitiveBufferMutated;
	private ComputeBuffer optimStepGradientsBuffer;
	private ComputeBuffer optimStepMutationError;
	private ComputeBuffer gradientMoments1Buffer;
	private ComputeBuffer gradientMoments2Buffer;
	private ComputeBuffer optimStepCounterBuffer;
	private ComputeBuffer primitiveKillCounters;
	private ComputeBuffer appendValidIDsBuffer;
	private ComputeBuffer appendInvalidIDsBuffer;
	private ComputeBuffer argsValidIDsBuffer;
	private ComputeBuffer argsInvalidIDsBuffer;
	private ComputeBuffer argsResampling;
	private GraphicsBuffer sortedPrimitiveIDBuffer;
	private ComputeBuffer sortedValidPrimitiveIDBuffer;
	private Bounds mesh3DSceneBounds;
	private int currentViewPoint = 0;
	public int currentOptimStep = 0;
	private int currentStochasticFrame = 0;
	private Color depthIDClearColor = new Color(4294967295, 0, 0, 0);
	private Stopwatch systemTimer = new Stopwatch();
	private int optimStepsSeparateCount = 1;
	private int vertexCount;
	private int triangleCount;

	// Compute kernels
	private int kernelReset;
	private int kernelRandomPerturbation;
	private int kernelGradientEstimation;
	private int kernelGradientEstimationPost;
	private int kernelGradientDescent;

	private int kernelResetVisibilityCounter;
	private int kernelDecrementVisibilityCounter;
	private int kernelResetSeenPrimitivesVisibilityCounter;
	private int kernelListValidAndInvalidPrimitiveIDs;
	private int kernelInitArgsResampling;
	private int kernelInitBitonicSortValidPrimitives;
	private int kernelBitonicSortValidPrimitives;
	private int kernelPairResampling;

	// ======================= INTERFACE =======================
	public Vector2Int targetResolution = new Vector2Int(512, 512);
	// Su:Add an init mesh
	public GameObject init3DMesh;
	public GameObject target3DMesh;
	public int primitiveCount = 1;
	public float primitiveInitSize = 1.0f;
	public int primitiveInitSeed = -1;
	public bool initPrimitivesOnMeshSurface = false;
	public Vector2 randomViewZoomRange = Vector2.one;

	public bool reset = false;
	public bool pause = false;
	public bool stepForward = false;
	public DisplayMode displayMode = DisplayMode.Optimization;
	public bool separateFreeViewCamera = true;

	public Optimizer optimizer = Optimizer.Adam;
	public LossMode lossMode = LossMode.L2;
	public bool doAlphaLoss = true;
	public int viewsPerOptimStep = 1;
	public bool optimizeColorsSeparately = false;

	[Range(0.0f, 1.0f)] public float beta1 = 0.9f;
	[Range(0.0f, 1.0f)] public float beta2 = 0.999f;
	[LogarithmicRange(0.0f, 0.001f, 1.0f)] public float learningRatePosition = 0.01f;
	[LogarithmicRange(0.0f, 0.001f, 1.0f)] public float learningRateColor = 0.01f;
	[LogarithmicRange(0.0f, 0.001f, 1.0f)] public float lambdaLap = 0.01f;

	public bool doPrimitiveResampling = true;
	public int resamplingInterval = 1;
	public int optimStepsUnseenBeforeKill = 16;
	public float minPrimitiveWorldArea = 0.0001f;

	public float millisecondsPerOptimStep = 0.0f;
	public float totalElapsedSeconds = 0.0f;
	
	
	

	// =========================== UNITY ===========================
	void OnEnable()
	{
		System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
		System.Globalization.CultureInfo.DefaultThreadCurrentUICulture = System.Globalization.CultureInfo.InvariantCulture;
		Physics.simulationMode = SimulationMode.Script;
		stochasticOptimizerCS = (ComputeShader)Resources.Load("StochasticOptimizer");
		triangleResamplingCS = (ComputeShader)Resources.Load("TriangleResampling");
		cameraDisplay = GameObject.Find("CameraDisplay").GetComponent<Camera>();
		cameraOptim = GameObject.Find("CameraOptim").GetComponent<Camera>();

		kernelReset = stochasticOptimizerCS.FindKernel("Reset");
		kernelRandomPerturbation = stochasticOptimizerCS.FindKernel("RandomPerturbation");
		kernelGradientEstimation = stochasticOptimizerCS.FindKernel("GradientEstimation");
		kernelGradientEstimationPost = stochasticOptimizerCS.FindKernel("GradientEstimationPost");
		kernelGradientDescent = stochasticOptimizerCS.FindKernel("GradientDescent");

		kernelResetVisibilityCounter = triangleResamplingCS.FindKernel("ResetVisibilityCounter");
		kernelDecrementVisibilityCounter = triangleResamplingCS.FindKernel("DecrementVisibilityCounter");
		kernelResetSeenPrimitivesVisibilityCounter = triangleResamplingCS.FindKernel("ResetSeenPrimitivesVisibilityCounter");
		kernelListValidAndInvalidPrimitiveIDs = triangleResamplingCS.FindKernel("ListValidAndInvalidPrimitiveIDs");
		kernelInitArgsResampling = triangleResamplingCS.FindKernel("InitArgsResampling");
		kernelInitBitonicSortValidPrimitives = triangleResamplingCS.FindKernel("InitBitonicSortValidPrimitives");
		kernelBitonicSortValidPrimitives = triangleResamplingCS.FindKernel("BitonicSortValidPrimitives");
		kernelPairResampling = triangleResamplingCS.FindKernel("PairResampling");

	}

	void Update()
	{
		// Keyboard controls
		if (Input.GetKeyDown(KeyCode.R)) reset = true;
		if (Input.GetKeyDown(KeyCode.P)) pause = !pause;
		if (Input.GetKeyDown(KeyCode.Space)) stepForward = true;
		if (Input.GetKeyDown(KeyCode.F)) separateFreeViewCamera = !separateFreeViewCamera;
		if (Input.GetKeyDown(KeyCode.F1)) displayMode = DisplayMode.Optimization;
		if (Input.GetKeyDown(KeyCode.F2)) displayMode = DisplayMode.Target;

		// First init
		if (currentOptimStep == 0)
		{
			ResetEverything();
			totalElapsedSeconds = 0.0f;
			systemTimer.Restart();
		}

		// Optimization Loop
		OptimizationUpdate();

		// Update display
		DisplayModeUpdate();
	}

	public void OptimizationUpdate()
	{
		if (pause == true && stepForward == false)
		{
			systemTimer.Restart();
			return;
		}

		// Prepare parameters
		SetSharedComputeFrameParameters(stochasticOptimizerCS);
		SetSharedComputeFrameParameters(triangleResamplingCS);

		// Decrement visibility counters every optim step
		if (doPrimitiveResampling == true)
		{
			triangleResamplingCS.SetBuffer(kernelDecrementVisibilityCounter, "_PrimitiveKillCounters", primitiveKillCounters);
			DispatchCompute1D(triangleResamplingCS, kernelDecrementVisibilityCounter, primitiveCount, 256);
		}

		// Accumulate gradients for this optim step
		for (int i = 0; i < viewsPerOptimStep; i++)
		{
			// Set up new view point
			stochasticOptimizerCS.SetInt("_CurrentView", currentViewPoint);
			RandomizeCameraView();
			cameraOptim.Render();

			// We do this next part twice if we want to optimize positions and colors separately, only once otherwise
			for (int j = 0; j < (optimizeColorsSeparately == true ? 2 : 1); j++)
			{
				if (optimizeColorsSeparately == true)
				{
					// Use the learning rate to disable position or color perturbation, as they're used as perturbation magnitude factors
					stochasticOptimizerCS.SetFloat("_LearningRatePosition", j == 0 ? learningRatePosition : 0.0f);
					stochasticOptimizerCS.SetFloat("_LearningRateColor", j == 1 ? learningRateColor : 0.0f);
				}

				// Minus Epsilon
				stochasticOptimizerCS.SetFloat("_IsAntitheticMutation", 1.0f);
				DoRandomPerturbation();
				RenderOptimizationSceneWithIDs(cameraOptim, primitiveBufferMutated, renderedFrameMutatedMinus, idBufferMutatedMinus);

				// Plus Epsilon
				stochasticOptimizerCS.SetFloat("_IsAntitheticMutation", 0.0f);
				DoRandomPerturbation();
				RenderOptimizationSceneWithIDs(cameraOptim, primitiveBufferMutated, renderedFrameMutatedPlus, idBufferMutatedPlus);

				// Accumulate Per-Pixel Stochastic Gradients
				DoGradientEstimation();
			}
			stochasticOptimizerCS.SetFloat("_LearningRatePosition", learningRatePosition);
			stochasticOptimizerCS.SetFloat("_LearningRateColor", learningRateColor);

			// Reset visibility counter of triangles seen during this step
			if (doPrimitiveResampling == true)
			{
				triangleResamplingCS.SetBuffer(kernelResetSeenPrimitivesVisibilityCounter, "_PrimitiveKillCounters", primitiveKillCounters);
				triangleResamplingCS.SetTexture(kernelResetSeenPrimitivesVisibilityCounter, "_IDBuffer", idBufferMutatedPlus);
				triangleResamplingCS.Dispatch(kernelResetSeenPrimitivesVisibilityCounter, (int)math.ceil(targetResolution.x / 16.0f), (int)math.ceil(targetResolution.y / 16.0f), 1);
			}

			// Next camera position
			currentViewPoint += 1;
		}


		// Apply accumulated gradient for this optim step
		DoGradientDescent();
		if (currentOptimStep > 1)
			DoTriangleResampling(0);
		ResetOptimizationStep();

		// Metrics
		currentOptimStep += 1;
		millisecondsPerOptimStep = (float)systemTimer.Elapsed.TotalMilliseconds;
		totalElapsedSeconds += (float)systemTimer.Elapsed.TotalMilliseconds / 1000.0f;
		systemTimer.Restart();

		if (stepForward == true)
			stepForward = false;
	}

	public void DisplayModeUpdate()
	{
		// Update if changed
		if (separateFreeViewCamera == false)
		{
			cameraDisplay.cullingMask = 0;
			if (displayMode == DisplayMode.Optimization)
			{
				cameraDisplay.GetComponent<DisplayRenderTexture>().displayRenderTexture = renderedFrameMutatedPlus;
			}
			else if (displayMode == DisplayMode.Target)
			{
				cameraDisplay.GetComponent<DisplayRenderTexture>().displayRenderTexture = targetFrameBuffer;
			}
		}
		else
		{
			if (displayMode == DisplayMode.Optimization)
			{
				cameraDisplay.GetComponent<DisplayRenderTexture>().displayRenderTexture = null;
				cameraDisplay.cullingMask = 0;
			}
			else if (displayMode == DisplayMode.Target)
			{
				cameraDisplay.GetComponent<DisplayRenderTexture>().displayRenderTexture = null;
				cameraDisplay.cullingMask = ~0;
			}
		}
	}

	void OnDisable()
	{
		ReleaseEverything();
	}

	void ReleaseEverything()
	{
		if (rasterMaterial != null)
		{
			if (Application.isPlaying)
			{
				Destroy(rasterMaterial);
			}
			else
			{
				DestroyImmediate(rasterMaterial);
			}
		}

		if (primitiveBuffer != null)
			primitiveBuffer.Release();
		primitiveBuffer = null;
		if (indexBuffer != null)
			indexBuffer.Release();
		indexBuffer = null;
		ReleaseOptimBuffers();
	}

	void ReleaseOptimBuffers()
	{
		if (primitiveBufferMutated != null)
			primitiveBufferMutated.Release();
		if (idBufferMutatedPlus != null)
			idBufferMutatedPlus.Release();
		if (idBufferMutatedMinus != null)
			idBufferMutatedMinus.Release();
		if (renderedFrameFreeView != null)
			renderedFrameFreeView.Release();
		if (renderedFrameMutatedMinus != null)
			renderedFrameMutatedMinus.Release();
		if (renderedFrameMutatedPlus != null)
			renderedFrameMutatedPlus.Release();
		if (optimStepGradientsBuffer != null)
			optimStepGradientsBuffer.Release();
		if (optimStepMutationError != null)
			optimStepMutationError.Release();
		if (gradientMoments1Buffer != null)
			gradientMoments1Buffer.Release();
		if (gradientMoments2Buffer != null)
			gradientMoments2Buffer.Release();
		if (optimStepCounterBuffer != null)
			optimStepCounterBuffer.Release();
		if (primitiveKillCounters != null)
			primitiveKillCounters.Release();
		if (appendValidIDsBuffer != null)
			appendValidIDsBuffer.Release();
		if (appendInvalidIDsBuffer != null)
			appendInvalidIDsBuffer.Release();
		if (argsValidIDsBuffer != null)
			argsValidIDsBuffer.Release();
		if (argsInvalidIDsBuffer != null)
			argsInvalidIDsBuffer.Release();
		if (sortedPrimitiveIDBuffer != null)
			sortedPrimitiveIDBuffer.Release();
		if (sortedValidPrimitiveIDBuffer != null)
			sortedValidPrimitiveIDBuffer.Release();
		if (targetFrameBuffer != null)
			targetFrameBuffer.Release();
		if (argsResampling != null)
			argsResampling.Release();
		if (AdjacencyBuffer != null)
			AdjacencyBuffer.Release();
		if (AdjacencyStartBuffer != null)
			AdjacencyStartBuffer.Release();
		if (AdjacencyCountBuffer != null)
			AdjacencyCountBuffer.Release();
	}
	
	void OnRenderObject()
	{
		// Only to render primitives in free view
		if (rasterMaterial == null || Camera.current == cameraOptim || displayMode != DisplayMode.Optimization || separateFreeViewCamera == false)
			return;

		Vector3 cameraPos = Camera.current.transform.position;
		Matrix4x4 cameraVP = GL.GetGPUProjectionMatrix(Camera.current.projectionMatrix, true) * Camera.current.worldToCameraMatrix;
		rasterMaterial.SetMatrix("_CameraMatrixVP", cameraVP);
		rasterMaterial.SetInt("_VertexCount", vertexCount);
		rasterMaterial.SetBuffer("_PrimitiveBuffer", primitiveBuffer);
		rasterMaterial.SetBuffer("_IndexBuffer", indexBuffer);
		// Enable debug wireframe mode
		rasterMaterial.SetInt("_EnableWireframe", 1);
		rasterMaterial.SetFloat("_WireWidth", 2.0f);
		rasterMaterial.SetPass(0);
		Graphics.DrawProceduralNow(MeshTopology.Triangles, triangleCount * 3);
	}




	// ======================= OPTIMIZATION =======================
	// Su: 
	public void RenderOptimizationSceneWithIDs(Camera cameraToUse, ComputeBuffer primitiveBufferToUse, RenderTexture renderTargetToUse, RenderTexture idRenderTargetToUse)
	{
		// Disable debug wireframe mode
		rasterMaterial.SetInt("_EnableWireframe", 0);
		//rasterMaterial.SetFloat("_WireWidth", 2.0f);
		
		// Camera setup
		Matrix4x4 cameraVP = GL.GetGPUProjectionMatrix(cameraToUse.projectionMatrix, true) * cameraToUse.worldToCameraMatrix;
		rasterMaterial.SetMatrix("_CameraMatrixVP", cameraVP);

		// Render primitives ID+Depth
		rasterMaterial.SetInt("_VertexCount", vertexCount);
		rasterMaterial.SetBuffer("_PrimitiveBuffer", primitiveBufferToUse);
		rasterMaterial.SetBuffer("_IndexBuffer", indexBuffer);

		// Clear targets
		Graphics.SetRenderTarget(idRenderTargetToUse.colorBuffer, idRenderTargetToUse.depthBuffer);
		GL.Clear(true, true, depthIDClearColor, 1.0f);
		Graphics.SetRenderTarget(renderTargetToUse.colorBuffer, renderTargetToUse.depthBuffer);
		GL.Clear(true, true, Color.clear, 1.0f);

		// Render both color + ID
		RenderBuffer[] mrt = new RenderBuffer[] { renderTargetToUse.colorBuffer, idRenderTargetToUse.colorBuffer };
		Graphics.SetRenderTarget(mrt, idRenderTargetToUse.depthBuffer);
		rasterMaterial.SetPass(0);
		Graphics.DrawProceduralNow(MeshTopology.Triangles, triangleCount * 3);
	}

	public void ResetOptimizationStep()
	{
		stochasticOptimizerCS.SetBuffer(kernelReset, "_PrimitiveGradientsOptimStep", optimStepGradientsBuffer);
		stochasticOptimizerCS.SetBuffer(kernelReset, "_PrimitiveMutationError", optimStepMutationError);
		DispatchCompute1D(stochasticOptimizerCS, kernelReset, primitiveBuffer.count, 256);
	}

	public void DoGradientEstimation()
	{
		// Accumulate per pixel image loss
		stochasticOptimizerCS.SetTexture(kernelGradientEstimation, "_TargetTexture", targetFrameBuffer);
		stochasticOptimizerCS.SetTexture(kernelGradientEstimation, "_ResolvedFrameMutatedMinus", renderedFrameMutatedMinus);
		stochasticOptimizerCS.SetTexture(kernelGradientEstimation, "_ResolvedFrameMutatedPlus", renderedFrameMutatedPlus);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimation, "_PrimitiveMutationError", optimStepMutationError);
		stochasticOptimizerCS.SetTexture(kernelGradientEstimation, "_IDBufferMutatedMinus", idBufferMutatedMinus);
		stochasticOptimizerCS.SetTexture(kernelGradientEstimation, "_IDBufferMutatedPlus", idBufferMutatedPlus);
		stochasticOptimizerCS.Dispatch(kernelGradientEstimation, (int)math.ceil(targetResolution.x / 16.0f), (int)math.ceil(targetResolution.y / 16.0f), 1);

		// Accumulate gradients
		stochasticOptimizerCS.SetFloat("_LambdaLap", lambdaLap);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_IndexBuffer", indexBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_AdjacencyBuffer", indexBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_AdjacencyStartBuffer", indexBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_AdjacencyCountBuffer", indexBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_PrimitiveBuffer", primitiveBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_PrimitiveBufferMutated", primitiveBufferMutated);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_PrimitiveMutationError", optimStepMutationError);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_PrimitiveGradientsOptimStep", optimStepGradientsBuffer);
		DispatchCompute1D(stochasticOptimizerCS, kernelGradientEstimationPost, primitiveCount, 256);
	}

	public void DoGradientDescent()
	{
		stochasticOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveBuffer", primitiveBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveGradientsMoments1", gradientMoments1Buffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveGradientsMoments2", gradientMoments2Buffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveGradientsOptimStep", optimStepGradientsBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveOptimStepCounter", optimStepCounterBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveMutationError", optimStepMutationError);
		DispatchCompute1D(stochasticOptimizerCS, kernelGradientDescent, primitiveBuffer.count, 256);
	}

	public void DoRandomPerturbation()
	{
		stochasticOptimizerCS.SetBuffer(kernelRandomPerturbation, "_PrimitiveBuffer", primitiveBuffer);
		stochasticOptimizerCS.SetBuffer(kernelRandomPerturbation, "_PrimitiveBufferMutated", primitiveBufferMutated);
		DispatchCompute1D(stochasticOptimizerCS, kernelRandomPerturbation, primitiveBuffer.count, 256);
	}

	public void DoTriangleResampling(int primitiveGroupToUse)
	{
		if (doPrimitiveResampling == false)
			return;

		// Only perform resampling at desired interval
		if (currentOptimStep % resamplingInterval != 0)
			return;

		// List valid and invalid primitive IDs
		appendValidIDsBuffer.SetCounterValue(0);
		appendInvalidIDsBuffer.SetCounterValue(0);
		triangleResamplingCS.SetInt("_CurrentPairingOffset", (int)(UnityEngine.Random.value * primitiveCount));
		triangleResamplingCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_PrimitiveBuffer", primitiveBuffer);
		triangleResamplingCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_PrimitiveGradientsMoments1", gradientMoments1Buffer);
		triangleResamplingCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_PrimitiveGradientsMoments2", gradientMoments2Buffer);
		triangleResamplingCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_PrimitiveOptimStepCounter", optimStepCounterBuffer);
		triangleResamplingCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_PrimitiveKillCounters", primitiveKillCounters);
		triangleResamplingCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_AppendValidPrimitiveIDs", appendValidIDsBuffer);
		triangleResamplingCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_AppendInvalidPrimitiveIDs", appendInvalidIDsBuffer);
		DispatchCompute1D(triangleResamplingCS, kernelListValidAndInvalidPrimitiveIDs, primitiveCount, 256);

		// Init resampling indirect dispatch args
		ComputeBuffer.CopyCount(appendValidIDsBuffer, argsValidIDsBuffer, 0);
		ComputeBuffer.CopyCount(appendInvalidIDsBuffer, argsInvalidIDsBuffer, 0);
		triangleResamplingCS.SetBuffer(kernelInitArgsResampling, "_ArgsValidPrimitiveIDs", argsValidIDsBuffer);
		triangleResamplingCS.SetBuffer(kernelInitArgsResampling, "_ArgsInvalidPrimitiveIDs", argsInvalidIDsBuffer);
		triangleResamplingCS.SetBuffer(kernelInitArgsResampling, "_ArgsResampling", argsResampling);
		triangleResamplingCS.Dispatch(kernelInitArgsResampling, 1, 1, 1);

		// Sort valid primitives by importance criteria
		triangleResamplingCS.SetInt("_PrimitiveCountPow2", sortedValidPrimitiveIDBuffer.count);
		triangleResamplingCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_PrimitiveBuffer", primitiveBuffer);
		triangleResamplingCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_PrimitiveGradientsMoments1", gradientMoments1Buffer);
		triangleResamplingCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_PrimitiveGradientsMoments2", gradientMoments2Buffer);
		triangleResamplingCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_ReadValidPrimitiveIDs", appendValidIDsBuffer);
		triangleResamplingCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_SortedValidPrimitiveIDs", sortedValidPrimitiveIDBuffer);
		triangleResamplingCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_ArgsValidPrimitiveIDs", argsValidIDsBuffer);
		DispatchCompute1D(triangleResamplingCS, kernelInitBitonicSortValidPrimitives, primitiveCount, 256);
		triangleResamplingCS.SetBuffer(kernelBitonicSortValidPrimitives, "_SortedValidPrimitiveIDs", sortedValidPrimitiveIDBuffer);
		for (uint d2 = 1; d2 < sortedValidPrimitiveIDBuffer.count; d2 *= 2)
		{
			for (uint d1 = d2; d1 >= 1u; d1 /= 2)
			{
				triangleResamplingCS.SetInt("_SortLoopValueX", (int)d1);
				triangleResamplingCS.SetInt("_SortLoopValueY", (int)d2);
				DispatchCompute1D(triangleResamplingCS, kernelBitonicSortValidPrimitives, sortedValidPrimitiveIDBuffer.count, 256);
			}
		}

		// Resample primitive pairs
		triangleResamplingCS.SetInt("_CurrentView", currentViewPoint);
		triangleResamplingCS.SetBuffer(kernelPairResampling, "_PrimitiveBuffer", primitiveBuffer);
		triangleResamplingCS.SetBuffer(kernelPairResampling, "_PrimitiveKillCounters", primitiveKillCounters);
		triangleResamplingCS.SetBuffer(kernelPairResampling, "_PrimitiveGradientsMoments1", gradientMoments1Buffer);
		triangleResamplingCS.SetBuffer(kernelPairResampling, "_PrimitiveGradientsMoments2", gradientMoments2Buffer);
		triangleResamplingCS.SetBuffer(kernelPairResampling, "_PrimitiveOptimStepCounter", optimStepCounterBuffer);
		triangleResamplingCS.SetBuffer(kernelPairResampling, "_ArgsResampling", argsResampling);
		triangleResamplingCS.SetBuffer(kernelPairResampling, "_ReadValidPrimitiveIDs", appendValidIDsBuffer);
		triangleResamplingCS.SetBuffer(kernelPairResampling, "_SortedValidPrimitiveIDs", sortedValidPrimitiveIDBuffer);
		triangleResamplingCS.SetBuffer(kernelPairResampling, "_ReadInvalidPrimitiveIDs", appendInvalidIDsBuffer);
		triangleResamplingCS.DispatchIndirect(kernelPairResampling, argsResampling);
	}




	// ======================== MANAGEMENT ========================
	public void SetSharedComputeFrameParameters(ComputeShader computeShader)
	{
		// Set shared parameters
		computeShader.SetInt("_VertexCount", vertexCount);
		computeShader.SetInt("_PrimitiveCount", primitiveCount);
		computeShader.SetInt("_CurrentOptimStep", currentOptimStep);
		computeShader.SetInt("_OutputWidth", targetResolution.x);
		computeShader.SetInt("_OutputHeight", targetResolution.y);
		computeShader.SetInt("_OptimizerMode", optimizer == Optimizer.RMSProp ? 0 : 1);
		computeShader.SetInt("_LossMode", lossMode == LossMode.L1 ? 0 : 1);
		computeShader.SetFloat("_OptimizerBeta1", beta1);
		computeShader.SetFloat("_OptimizerBeta2", beta2);
		computeShader.SetFloat("_MinPrimitiveWorldArea", minPrimitiveWorldArea);
		computeShader.SetInt("_FramesUnseenBeforeKill", optimStepsUnseenBeforeKill);
		computeShader.SetInt("_ViewsPerOptimStep", viewsPerOptimStep);
		computeShader.SetFloat("_DoAlphaLoss", doAlphaLoss ? 1.0f : 0.0f);
		computeShader.SetFloat("_LearningRatePosition", learningRatePosition);
		computeShader.SetFloat("_LearningRateColor", learningRateColor);
	}

	public void RandomizeCameraView()
	{
		Bounds targetBounds = mesh3DSceneBounds;
		float distance = targetBounds.extents.magnitude;
		distance *= (randomViewZoomRange.x + UnityEngine.Random.value * (randomViewZoomRange.y - randomViewZoomRange.x));
		distance *= 2;
		cameraOptim.transform.position = targetBounds.center + UnityEngine.Random.onUnitSphere * distance;
		float3 lookAtCenter = new float3(targetBounds.center) + (new float3(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value) * 2.0f - 1.0f) * targetBounds.extents * 0.5f;
		cameraOptim.transform.LookAt(lookAtCenter, UnityEngine.Random.onUnitSphere);
	}

	public void ResetEverything()
	{
		OnDisable();
		OnEnable();

		target3DMesh.SetActive(true);
		mesh3DSceneBounds = new Bounds();
		MeshRenderer[] allRenderers = FindObjectsOfType<MeshRenderer>();
		for (int i = 0; i < allRenderers.Length; i++)
			if (allRenderers[i].gameObject.activeInHierarchy == true)
				mesh3DSceneBounds.Encapsulate(allRenderers[i].bounds);

		// Primitive buffer
		// InitPrimitiveBuffer();
		// Su: Use an init mesh to init the primitive buffer
		InitTrianglePrimitiveBufferWithInitMesh(ref primitiveBuffer);

		cameraDisplay.GetComponent<OrbitCamera>().target = mesh3DSceneBounds;
		InitAllOptimBuffers();
		ResetOptimizationStep();

		// Set up 3D view camera
		cameraOptim.enabled = false;
		cameraOptim.targetTexture = targetFrameBuffer;
		cameraDisplay.orthographic = false;

		currentOptimStep = 0;
		currentViewPoint = 0;
		totalElapsedSeconds = 0.0f;
		systemTimer.Restart();
	}

	public void InitAllOptimBuffers()
	{
		ReleaseOptimBuffers();

		// Init rendering material
		rasterMaterial = new Material(Shader.Find("Custom/MeshRasterizer"));
		rasterMaterial.hideFlags = HideFlags.DontSave;

		// Init everything
		idBufferMutatedPlus = new RenderTexture(targetResolution.x, targetResolution.y, 32, RenderTextureFormat.RFloat, RenderTextureReadWrite.Linear);
		idBufferMutatedPlus.enableRandomWrite = true;
		idBufferMutatedMinus = new RenderTexture(targetResolution.x, targetResolution.y, 32, RenderTextureFormat.RFloat, RenderTextureReadWrite.Linear);
		idBufferMutatedMinus.enableRandomWrite = true;
		renderedFrameMutatedMinus = new RenderTexture(targetResolution.x, targetResolution.y, 32, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
		renderedFrameMutatedMinus.enableRandomWrite = true;
		renderedFrameMutatedPlus = new RenderTexture(targetResolution.x, targetResolution.y, 32, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
		renderedFrameMutatedPlus.enableRandomWrite = true;
		renderedFrameFreeView = new RenderTexture(targetResolution.x, targetResolution.y, 32, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
		renderedFrameFreeView.enableRandomWrite = true;
		targetFrameBuffer = new RenderTexture(targetResolution.x, targetResolution.y, 32, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);

		// Optim buffers
		int primitiveByteSize = 12;
		optimStepGradientsBuffer = new ComputeBuffer(primitiveCount, sizeof(int) * 3);
		gradientMoments1Buffer = new ComputeBuffer(primitiveCount, primitiveByteSize);
		gradientMoments2Buffer = new ComputeBuffer(primitiveCount, primitiveByteSize);
		primitiveBufferMutated = new ComputeBuffer(primitiveCount, primitiveByteSize);
		optimStepMutationError = new ComputeBuffer(triangleCount, sizeof(int));
		optimStepCounterBuffer = new ComputeBuffer(primitiveCount, sizeof(int));
		primitiveKillCounters = new ComputeBuffer(primitiveCount, sizeof(int));
		ZeroInitBuffer(optimStepGradientsBuffer);
		ZeroInitBuffer(gradientMoments1Buffer);
		ZeroInitBuffer(gradientMoments2Buffer);
		ZeroInitBuffer(primitiveBufferMutated);
		ZeroInitBuffer(optimStepMutationError);
		ZeroInitBuffer(optimStepCounterBuffer);

		// Resampling buffers
		appendValidIDsBuffer = new ComputeBuffer(primitiveCount, sizeof(uint), ComputeBufferType.Append);
		argsValidIDsBuffer = new ComputeBuffer(4, sizeof(int), ComputeBufferType.IndirectArguments);
		appendInvalidIDsBuffer = new ComputeBuffer(primitiveCount, sizeof(uint), ComputeBufferType.Append);
		argsInvalidIDsBuffer = new ComputeBuffer(4, sizeof(int), ComputeBufferType.IndirectArguments);
		argsResampling = new ComputeBuffer(4, sizeof(int), ComputeBufferType.IndirectArguments);
		int sortCount = (int)(math.ceilpow2(primitiveCount));
		sortedValidPrimitiveIDBuffer = new ComputeBuffer(sortCount, sizeof(uint) * 2);

		// Init kill counters
		triangleResamplingCS.SetInt("_PrimitiveCount", primitiveBuffer.count);
		triangleResamplingCS.SetInt("_FramesUnseenBeforeKill", optimStepsUnseenBeforeKill);
		triangleResamplingCS.SetBuffer(kernelResetVisibilityCounter, "_PrimitiveKillCounters", primitiveKillCounters);
		DispatchCompute1D(triangleResamplingCS, kernelResetVisibilityCounter, primitiveBuffer.count, 256);
	}

	private static void DispatchCompute1D(ComputeShader compute, int kernel, int threadCount, int groupSizeX)
	{
		int threadGroupCount = (int)math.ceil(threadCount / (float)groupSizeX);
		int dispatchCount = (int)math.ceil(threadGroupCount / 65536.0f);
		int offset = 0;
		for (int i = 0; i < dispatchCount; i++)
		{
			int currentGroupCount = (int)math.min(threadGroupCount - offset, 65535.0f);
			compute.SetInt("_Dispatch1DOffset", offset);
			compute.Dispatch(kernel, currentGroupCount, 1, 1);
			offset += currentGroupCount;
		}
	}

	public void ZeroInitBuffer(ComputeBuffer buffer)
	{
		byte[] temp = new byte[buffer.count * buffer.stride];
		buffer.SetData(temp);
	}




	// ======================= PRIMITIVE INIT =======================
	public void InitPrimitiveBuffer()
	{
		// Random seed
		if (primitiveInitSeed < 0)
			primitiveInitSeed = (int)(UnityEngine.Random.value * int.MaxValue);
		UnityEngine.Random.InitState(primitiveInitSeed);

		// Init random positions
		float3[] positions = new float3[primitiveCount];
		float[] initSizes = new float[0];

		if (initPrimitivesOnMeshSurface == true)
			InitPositionsOnMeshSurface(positions);
		else
			InitPositionsInsideBounds(positions);

		// Init primitive buffer
		InitTrianglePrimitiveBuffer(ref primitiveBuffer, positions, initSizes);
	}

	public void InitPositionsInsideBounds(float3[] positions)
	{
		// Init random primitives within target bounds
		Bounds targetBounds = mesh3DSceneBounds;
		for (int i = 0; i < primitiveCount; i++)
		{
			float3 randPos = new float3(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value) * 2.0f - 1.0f;
			randPos = new float3(targetBounds.center) + randPos * new float3(targetBounds.extents);
			positions[i] = randPos;
		}
	}

	public void InitPositionsOnMeshSurface(float3[] positions)
	{
		// Init random triangles on target mesh surface
		Vector3[] targetVertices = target3DMesh.GetComponent<MeshFilter>().sharedMesh.vertices;
		int[] triangles = target3DMesh.GetComponent<MeshFilter>().sharedMesh.triangles;
		int targetTriangleCount = triangles.Length / 3;

		for (int i = 0; i < primitiveCount; i++)
		{
			// Select random triangle
			int triangleID = (int)math.min(targetTriangleCount - 1, UnityEngine.Random.value * targetTriangleCount);
			Vector3 vertexA = target3DMesh.transform.TransformPoint(targetVertices[triangles[triangleID * 3 + 0]]);
			Vector3 vertexB = target3DMesh.transform.TransformPoint(targetVertices[triangles[triangleID * 3 + 1]]);
			Vector3 vertexC = target3DMesh.transform.TransformPoint(targetVertices[triangles[triangleID * 3 + 2]]);

			// Random position in triangle
			float2 randTriangle = new float2(UnityEngine.Random.value, UnityEngine.Random.value);
			if (randTriangle.x + randTriangle.y >= 1)
				randTriangle = 1.0f - randTriangle;
			float3 randPos = vertexA + randTriangle.x * (vertexB - vertexA) + randTriangle.y * (vertexC - vertexA);
			positions[i] = randPos;
		}
	}

	public void InitTrianglePrimitiveBuffer(ref ComputeBuffer primitiveBufferToInit, float3[] positions, float[] initSizes)
	{
		// Init data on CPU
		int primitiveFloatSize = 12;
		float[] randData = new float[primitiveCount * primitiveFloatSize];

		for (int i = 0; i < primitiveCount; i++)
		{
			int offset = 0;

			// Init random triangle positions
			float initSizeToUse = primitiveInitSize * (initSizes.Length > 0 ? initSizes[i] : 1);
			float3 randPos = positions[i];
			float3 position0 = randPos + new float3(UnityEngine.Random.onUnitSphere) * initSizeToUse;
			float3 position1 = randPos + new float3(UnityEngine.Random.onUnitSphere) * initSizeToUse;
			float3 position2 = randPos + new float3(UnityEngine.Random.onUnitSphere) * initSizeToUse;
			randData[i * primitiveFloatSize + offset + 0] = position0.x; randData[i * primitiveFloatSize + offset + 1] = position0.y; randData[i * primitiveFloatSize + offset + 2] = position0.z; offset += 3;
			randData[i * primitiveFloatSize + offset + 0] = position1.x; randData[i * primitiveFloatSize + offset + 1] = position1.y; randData[i * primitiveFloatSize + offset + 2] = position1.z; offset += 3;
			randData[i * primitiveFloatSize + offset + 0] = position2.x; randData[i * primitiveFloatSize + offset + 1] = position2.y; randData[i * primitiveFloatSize + offset + 2] = position2.z; offset += 3;

			// Init random triangle color
			Color randColor = UnityEngine.Random.ColorHSV(0, 1, 0, 1);
			float3 color = new float3(randColor.r, randColor.g, randColor.b);
			randData[i * primitiveFloatSize + offset + 0] = color.x; randData[i * primitiveFloatSize + offset + 1] = color.y; randData[i * primitiveFloatSize + offset + 2] = color.z; offset += 3;
		}

		// Upload data to GPU
		primitiveBufferToInit = new ComputeBuffer(primitiveCount, sizeof(float) * primitiveFloatSize);
		primitiveBufferToInit.SetData(randData);
	}

	// Su:Init triangle primitive buffer with an init mesh
	public void InitTrianglePrimitiveBufferWithInitMesh(ref ComputeBuffer primitiveBufferToInit)
	{
		// Init data on CPU
		// Get the init mesh's geometry data
		//Vector3[] vertices = init3DMesh.GetComponent<MeshFilter>().sharedMesh.vertices;
		// The "triangles" below is actually the index buffer
		//int[] triangles = init3DMesh.GetComponent<MeshFilter>().sharedMesh.triangles;
		Vector3[] vertices;
		int[] triangles;

		IcosphereGenerator.Generate(
			subdivision: 4,
			out vertices,
			out triangles
			);

		triangleCount = triangles.Length / 3;
		indexBuffer = new ComputeBuffer(triangles.Length, sizeof(int));
		indexBuffer.SetData(triangles);
		vertexCount = vertices.Length;
		
		int[] adjacency;
		int[] adjacencyStart;
		int[] adjacencyCount;
		
		IcosphereGenerator.GetAdjacency(
			vertices,
			triangles,
			out adjacency,
			out adjacencyStart,
			out adjacencyCount
		);
		
		AdjacencyBuffer = new ComputeBuffer(adjacency.Length, sizeof(int));
		AdjacencyBuffer.SetData(adjacency);
		AdjacencyStartBuffer =  new ComputeBuffer(adjacencyStart.Length, sizeof(int));
		AdjacencyStartBuffer.SetData(adjacencyStart);
		AdjacencyCountBuffer = new ComputeBuffer(adjacencyCount.Length, sizeof(int));
		AdjacencyCountBuffer.SetData(adjacencyCount);
		
		// Primitive Buffer: vertex(XYZ) + triangle color(RGB)
		primitiveCount = vertexCount + triangleCount;
		float3[] data = new float3[primitiveCount];

		for (int i = 0; i < vertexCount; i++)
		{
			Vector3 vertexWorldPosition = init3DMesh.transform.TransformPoint(vertices[i]);
			data[i] = vertexWorldPosition;
		}

		for (int i = vertexCount; i < primitiveCount; i++)
		{
			//Color initColor = UnityEngine.Random.ColorHSV(0, 1, 0, 1);
			Color initColor = Color.blue;
			data[i] = new float3(initColor.r, initColor.g, initColor.b);
		}
		
		// Upload data to GPU
		primitiveBufferToInit = new ComputeBuffer(primitiveCount, sizeof(float) * 3);
		primitiveBufferToInit.SetData(data);
	}


	// ======================= STUFF =======================
	public enum DisplayMode
	{
		Optimization,
		Target
	}

	public enum LossMode
	{
		L1,
		L2
	}

	public enum Optimizer
	{
		RMSProp,
		Adam
	}
}
