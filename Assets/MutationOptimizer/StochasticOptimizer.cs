using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using DataStructures.ViliWonka.KDTree;
#if UNITY_EDITOR
using Unity.Burst;
#endif
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Profiling;
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
	private ComputeBuffer jacobiGradientsInBuffer;
	private ComputeBuffer jacobiGradientsOutBuffer;
	private ComputeBuffer optimStepMutationError;
	private ComputeBuffer gradientMoments1Buffer;
	private ComputeBuffer gradientMoments2Buffer;
	private ComputeBuffer optimStepCounterBuffer;
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
	private int kernelInitJacobiGradient;
	private int kernelGradientPrecondition;
	private int kernelGradientDescent;

	// ======================= INTERFACE =======================
	public Vector2Int targetResolution = new Vector2Int(512, 512);
	// Su:Add an init mesh
	public GameObject init3DMesh;
	public GameObject target3DMesh;
	public int primitiveCount;
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

	public float millisecondsPerOptimStep = 0.0f;
	public float totalElapsedSeconds = 0.0f;

	public bool visualizeLoss;
	private float chamfer;
	private Vector3[] verticesDst;
	private KDTree verticesDstKDTree;
	private KDQuery query;
	

	// =========================== UNITY ===========================
	void OnEnable()
	{
		System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
		System.Globalization.CultureInfo.DefaultThreadCurrentUICulture = System.Globalization.CultureInfo.InvariantCulture;
		Physics.simulationMode = SimulationMode.Script;
		stochasticOptimizerCS = (ComputeShader)Resources.Load("StochasticOptimizer");
		cameraDisplay = GameObject.Find("CameraDisplay").GetComponent<Camera>();
		cameraOptim = GameObject.Find("CameraOptim").GetComponent<Camera>();

		kernelReset = stochasticOptimizerCS.FindKernel("Reset");
		kernelRandomPerturbation = stochasticOptimizerCS.FindKernel("RandomPerturbation");
		kernelGradientEstimation = stochasticOptimizerCS.FindKernel("GradientEstimation");
		kernelGradientEstimationPost = stochasticOptimizerCS.FindKernel("GradientEstimationPost");
		kernelInitJacobiGradient = stochasticOptimizerCS.FindKernel("InitJacobiGradient");
		kernelGradientPrecondition = stochasticOptimizerCS.FindKernel("GradientPrecondition");
		kernelGradientDescent = stochasticOptimizerCS.FindKernel("GradientDescent");
		
		if (visualizeLoss)
			DebugGUI.SetGraphProperties("chamfer", "chamfer", 0, 0.5f, 1, Color.red, true);
		
		verticesDst = target3DMesh.GetComponentInChildren<MeshFilter>().sharedMesh.vertices;
		for (int i = 0; i < verticesDst.Length; i++)
		{
			verticesDst[i] = target3DMesh.transform.TransformPoint(verticesDst[i]);
		}
		verticesDstKDTree = new KDTree(verticesDst);
		query =  new KDQuery();
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

			// Next camera position
			currentViewPoint += 1;
		}


		// Apply accumulated gradient for this optim step
		DoGradientDescent();
		ResetOptimizationStep();
		
		Profiler.BeginSample("VisualizeLoss");
		// Visualize loss
		if (visualizeLoss && currentOptimStep % 50 == 0)
			ComputeLoss();
		Profiler.EndSample();

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
		if (AdjacencyBuffer != null)
			AdjacencyBuffer.Release();
		AdjacencyBuffer = null;
		if (AdjacencyStartBuffer != null)
			AdjacencyStartBuffer.Release();
		AdjacencyStartBuffer = null;
		if (AdjacencyCountBuffer != null)
			AdjacencyCountBuffer.Release();
		AdjacencyCountBuffer = null;
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
		if (jacobiGradientsInBuffer != null)
			jacobiGradientsInBuffer.Release();
		if (jacobiGradientsOutBuffer != null)
			jacobiGradientsOutBuffer.Release();
		if (optimStepMutationError != null)
			optimStepMutationError.Release();
		if (gradientMoments1Buffer != null)
			gradientMoments1Buffer.Release();
		if (gradientMoments2Buffer != null)
			gradientMoments2Buffer.Release();
		if (optimStepCounterBuffer != null)
			optimStepCounterBuffer.Release();
		if (targetFrameBuffer != null)
			targetFrameBuffer.Release();
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
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_AdjacencyBuffer", AdjacencyBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_AdjacencyStartBuffer", AdjacencyStartBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_AdjacencyCountBuffer", AdjacencyCountBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_PrimitiveBuffer", primitiveBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_PrimitiveBufferMutated", primitiveBufferMutated);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_PrimitiveMutationError", optimStepMutationError);
		stochasticOptimizerCS.SetBuffer(kernelGradientEstimationPost, "_PrimitiveGradientsOptimStep", optimStepGradientsBuffer);
		DispatchCompute1D(stochasticOptimizerCS, kernelGradientEstimationPost, primitiveCount, 256);
		
		// Init ping-pong buffer
		stochasticOptimizerCS.SetBuffer(kernelInitJacobiGradient, "_PrimitiveGradientsOptimStep", optimStepGradientsBuffer);
		stochasticOptimizerCS.SetBuffer(kernelInitJacobiGradient, "_JacobiGradientsIn", jacobiGradientsInBuffer);
		DispatchCompute1D(stochasticOptimizerCS, kernelInitJacobiGradient, primitiveCount, 256);
		
		// Gradient Precondition
		stochasticOptimizerCS.SetFloat("_Alpha", 0.002f);
		stochasticOptimizerCS.SetFloat("_LambdaLap", lambdaLap);
		stochasticOptimizerCS.SetBuffer(kernelGradientPrecondition, "_PrimitiveGradientsOptimStep", optimStepGradientsBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientPrecondition, "_AdjacencyBuffer", AdjacencyBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientPrecondition, "_AdjacencyStartBuffer", AdjacencyStartBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientPrecondition, "_AdjacencyCountBuffer", AdjacencyCountBuffer);
		for (int k = 0; k < 15; k++)
		{
			stochasticOptimizerCS.SetBuffer(kernelGradientPrecondition, "_JacobiGradientsIn", jacobiGradientsInBuffer);
			stochasticOptimizerCS.SetBuffer(kernelGradientPrecondition, "_JacobiGradientsOut", jacobiGradientsOutBuffer);
			DispatchCompute1D(stochasticOptimizerCS, kernelGradientPrecondition, vertexCount, 256);
			
			stochasticOptimizerCS.SetBuffer(kernelGradientPrecondition, "_JacobiGradientsIn", jacobiGradientsOutBuffer);
			stochasticOptimizerCS.SetBuffer(kernelGradientPrecondition, "_JacobiGradientsOut", jacobiGradientsInBuffer);
			DispatchCompute1D(stochasticOptimizerCS, kernelGradientPrecondition, vertexCount, 256);
		}
	}

	public void DoGradientDescent()
	{
		stochasticOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveBuffer", primitiveBuffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveGradientsMoments1", gradientMoments1Buffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveGradientsMoments2", gradientMoments2Buffer);
		stochasticOptimizerCS.SetBuffer(kernelGradientDescent, "_PrimitiveGradientsOptimStep", jacobiGradientsInBuffer);
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
	
	
	
	
	// ======================= EVALUATION =======================
	public void ComputeLoss()
	{
		Vector3[] verticesSrc = new Vector3[vertexCount];
		primitiveBuffer.GetData(verticesSrc, 0, 0, vertexCount);
		KDTree verticesSrcKDTree = new KDTree(verticesSrc);
		
		// Compute distance
		double sumSrcToDst = 0.0;
		double sumDstToSrc = 0.0;

		// ----- SRC -> DST -----
		for (int i = 0; i < verticesSrc.Length; i++)
		{
			Vector3 a = verticesSrc[i];
			
			List<float> distances = new List<float>();
			query.ClosestPoint(verticesDstKDTree, a,  null, distances);
			float minDistSqr = distances[0];

			float dist = Mathf.Sqrt(minDistSqr);

			sumSrcToDst += dist;
		}

		// ----- DST -> SRC -----
		for (int i = 0; i < verticesDst.Length; i++)
		{
			Vector3 b = verticesDst[i];
			
			List<float> distances = new List<float>();
			query.ClosestPoint(verticesSrcKDTree, b,  null, distances);
			float minDistSqr = distances[0];

			float dist = Mathf.Sqrt(minDistSqr);

			sumDstToSrc += dist;
		}
		
		chamfer = (float)(
			sumSrcToDst / verticesSrc.Length +
			sumDstToSrc / verticesDst.Length
		);
		
		DebugGUI.Graph("chamfer", chamfer);
		Debug.Log("step: " + currentOptimStep + " chamfer: " + chamfer);
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
		jacobiGradientsInBuffer = new ComputeBuffer(primitiveCount, sizeof(int) * 3);
		jacobiGradientsOutBuffer = new ComputeBuffer(primitiveCount, sizeof(int) * 3);
		gradientMoments1Buffer = new ComputeBuffer(primitiveCount, primitiveByteSize);
		gradientMoments2Buffer = new ComputeBuffer(primitiveCount, primitiveByteSize);
		primitiveBufferMutated = new ComputeBuffer(primitiveCount, primitiveByteSize);
		optimStepMutationError = new ComputeBuffer(triangleCount, sizeof(int));
		optimStepCounterBuffer = new ComputeBuffer(primitiveCount, sizeof(int));
		ZeroInitBuffer(optimStepGradientsBuffer);
		ZeroInitBuffer(jacobiGradientsInBuffer);
		ZeroInitBuffer(jacobiGradientsOutBuffer);
		ZeroInitBuffer(gradientMoments1Buffer);
		ZeroInitBuffer(gradientMoments2Buffer);
		ZeroInitBuffer(primitiveBufferMutated);
		ZeroInitBuffer(optimStepMutationError);
		ZeroInitBuffer(optimStepCounterBuffer);
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
