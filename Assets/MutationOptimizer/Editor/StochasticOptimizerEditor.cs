using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using Unity.Mathematics;


[CustomEditor(typeof(StochasticOptimizer))]
public class StochasticOptimizerEditor : Editor
{
	SerializedProperty targetResolution;
	SerializedProperty init3DMesh;
	SerializedProperty target3DMesh;
	SerializedProperty randomViewZoomRange;
	SerializedProperty initialMeshType;
	SerializedProperty icosphereSubdivision;
	SerializedProperty torusMajorSegments;
	SerializedProperty torusMinorSegments;
	SerializedProperty torusMajorRadius;
	SerializedProperty torusMinorRadius;

	SerializedProperty reset;
	SerializedProperty pause;
	SerializedProperty visualizeLoss;
	SerializedProperty step2;
	SerializedProperty stepForward;
	SerializedProperty displayMode;
	SerializedProperty separateFreeViewCamera;

	SerializedProperty optimizer;
	SerializedProperty lossMode;
	SerializedProperty viewMode;
	SerializedProperty regularizationMode;
	SerializedProperty doAlphaLoss;
	SerializedProperty viewsPerOptimStep;
	SerializedProperty optimizeColorsSeparately;
	SerializedProperty explicitLaplacianLambda;
	SerializedProperty gradientPreconditionLambda;

	SerializedProperty beta1;
	SerializedProperty beta2;
	SerializedProperty learningRatePosition;
	SerializedProperty learningRateColor;

	SerializedProperty millisecondsPerOptimStep;
	SerializedProperty totalElapsedSeconds;
	SerializedProperty textureMSE;
	SerializedProperty texturePSNR;
	SerializedProperty experimentMode;
	SerializedProperty targetLODLevel;
	SerializedProperty textureMSEUpdateInterval;

	void OnEnable()
	{
		targetResolution = serializedObject.FindProperty("targetResolution");
		init3DMesh = serializedObject.FindProperty("init3DMesh");
		target3DMesh = serializedObject.FindProperty("target3DMesh");
		randomViewZoomRange = serializedObject.FindProperty("randomViewZoomRange");
		initialMeshType = serializedObject.FindProperty("initialMeshType");
		icosphereSubdivision = serializedObject.FindProperty("icosphereSubdivision");
		torusMajorSegments = serializedObject.FindProperty("torusMajorSegments");
		torusMinorSegments = serializedObject.FindProperty("torusMinorSegments");
		torusMajorRadius = serializedObject.FindProperty("torusMajorRadius");
		torusMinorRadius = serializedObject.FindProperty("torusMinorRadius");

		reset = serializedObject.FindProperty("reset");
		pause = serializedObject.FindProperty("pause");
		visualizeLoss = serializedObject.FindProperty("visualizeLoss");
		step2 = serializedObject.FindProperty("step2");
		stepForward = serializedObject.FindProperty("stepForward");
		displayMode = serializedObject.FindProperty("displayMode");
		separateFreeViewCamera = serializedObject.FindProperty("separateFreeViewCamera");

		optimizer = serializedObject.FindProperty("optimizer");
		lossMode = serializedObject.FindProperty("lossMode");
		viewMode = serializedObject.FindProperty("viewMode");
		regularizationMode = serializedObject.FindProperty("regularizationMode");
		doAlphaLoss = serializedObject.FindProperty("doAlphaLoss");
		viewsPerOptimStep = serializedObject.FindProperty("viewsPerOptimStep");
		optimizeColorsSeparately = serializedObject.FindProperty("optimizeColorsSeparately");
		explicitLaplacianLambda = serializedObject.FindProperty("explicitLaplacianLambda");
		gradientPreconditionLambda = serializedObject.FindProperty("gradientPreconditionLambda");

		beta1 = serializedObject.FindProperty("beta1");
		beta2 = serializedObject.FindProperty("beta2");
		learningRatePosition = serializedObject.FindProperty("learningRatePosition");
		learningRateColor = serializedObject.FindProperty("learningRateColor");

		totalElapsedSeconds = serializedObject.FindProperty("totalElapsedSeconds");
		millisecondsPerOptimStep = serializedObject.FindProperty("millisecondsPerOptimStep");
		textureMSE = serializedObject.FindProperty("textureMSE");
		texturePSNR = serializedObject.FindProperty("texturePSNR");
		experimentMode = serializedObject.FindProperty("experimentMode");
		targetLODLevel = serializedObject.FindProperty("targetLODLevel");
		textureMSEUpdateInterval = serializedObject.FindProperty("textureMSEUpdateInterval");
	}

	public void ValidateParameters()
	{
		// Validate parameters
		viewsPerOptimStep.intValue = math.max(viewsPerOptimStep.intValue, 1);
		targetLODLevel.intValue = math.max(targetLODLevel.intValue, 0);
		textureMSEUpdateInterval.intValue = math.max(textureMSEUpdateInterval.intValue, 1);
		icosphereSubdivision.intValue = math.max(icosphereSubdivision.intValue, 0);
		torusMajorSegments.intValue = math.max(torusMajorSegments.intValue, 3);
		torusMinorSegments.intValue = math.max(torusMinorSegments.intValue, 3);
		torusMajorRadius.floatValue = math.max(torusMajorRadius.floatValue, 0.0001f);
		torusMinorRadius.floatValue = math.max(torusMinorRadius.floatValue, 0.0001f);
		targetResolution.vector2IntValue = new Vector2Int(math.max(targetResolution.vector2IntValue.x, 2), math.max(targetResolution.vector2IntValue.y, 2));
	}

	public override void OnInspectorGUI()
	{
		serializedObject.Update();

		// Metrics display
		EditorGUILayout.LabelField("Metrics Display", EditorStyles.boldLabel);
		EditorGUI.BeginDisabledGroup(true);
		EditorGUILayout.PropertyField(totalElapsedSeconds);
		EditorGUILayout.PropertyField(millisecondsPerOptimStep);
		EditorGUILayout.PropertyField(textureMSE);
		EditorGUILayout.PropertyField(texturePSNR);
		EditorGUI.EndDisabledGroup();
		EditorGUILayout.Space();

		// Scene settings
		EditorGUILayout.LabelField("Scene Settings", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(targetResolution);
		EditorGUILayout.PropertyField(init3DMesh);
		EditorGUILayout.PropertyField(target3DMesh);
		EditorGUILayout.PropertyField(randomViewZoomRange);
		EditorGUILayout.PropertyField(initialMeshType);
		EditorGUILayout.PropertyField(icosphereSubdivision);
		EditorGUILayout.PropertyField(torusMajorSegments);
		EditorGUILayout.PropertyField(torusMinorSegments);
		EditorGUILayout.PropertyField(torusMajorRadius);
		EditorGUILayout.PropertyField(torusMinorRadius);
		EditorGUILayout.PropertyField(experimentMode);
		EditorGUILayout.PropertyField(targetLODLevel);
		EditorGUILayout.PropertyField(textureMSEUpdateInterval);

		// Controls
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Controls", EditorStyles.boldLabel);
		if (pause.boolValue == true)
			if (GUILayout.Button("Step Forward"))
				stepForward.boolValue = true;
		EditorGUILayout.PropertyField(pause);
		EditorGUILayout.PropertyField(visualizeLoss);
		EditorGUILayout.PropertyField(step2);
		EditorGUILayout.PropertyField(displayMode);
		EditorGUILayout.PropertyField(separateFreeViewCamera);

		// Optimizer settings
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Optimizer Settings", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(optimizer);
		EditorGUILayout.PropertyField(lossMode);
		EditorGUILayout.PropertyField(viewMode);
		EditorGUILayout.PropertyField(regularizationMode);
		EditorGUILayout.PropertyField(doAlphaLoss);
		EditorGUILayout.PropertyField(viewsPerOptimStep);
		EditorGUILayout.PropertyField(optimizeColorsSeparately);
		EditorGUILayout.PropertyField(explicitLaplacianLambda);
		EditorGUILayout.PropertyField(gradientPreconditionLambda);

		// Optimizer controls
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Optimizer Controls", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(beta1);
		EditorGUILayout.PropertyField(beta2);
		EditorGUILayout.PropertyField(learningRatePosition);
		EditorGUILayout.PropertyField(learningRateColor);

		ValidateParameters();

		serializedObject.ApplyModifiedProperties();
	}
}
