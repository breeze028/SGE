using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using Unity.Mathematics;


[CustomEditor(typeof(StochasticOptimizer))]
public class StochasticOptimizerEditor : Editor
{
	SerializedProperty targetResolution;
	SerializedProperty target3DMesh;
	SerializedProperty primitiveCount;
	SerializedProperty primitiveInitSize;
	SerializedProperty primitiveInitSeed;
	SerializedProperty initPrimitivesOnMeshSurface;
	SerializedProperty randomViewZoomRange;

	SerializedProperty reset;
	SerializedProperty pause;
	SerializedProperty stepForward;
	SerializedProperty displayMode;
	SerializedProperty separateFreeViewCamera;

	SerializedProperty optimizer;
	SerializedProperty lossMode;
	SerializedProperty doAlphaLoss;
	SerializedProperty viewsPerOptimStep;
	SerializedProperty optimizeColorsSeparately;

	SerializedProperty beta1;
	SerializedProperty beta2;
	SerializedProperty learningRatePosition;
	SerializedProperty learningRateColor;

	SerializedProperty doPrimitiveResampling;
	SerializedProperty resamplingInterval;
	SerializedProperty optimStepsUnseenBeforeKill;
	SerializedProperty minPrimitiveWorldArea;

	SerializedProperty millisecondsPerOptimStep;
	SerializedProperty totalElapsedSeconds;

	void OnEnable()
	{
		targetResolution = serializedObject.FindProperty("targetResolution");
		target3DMesh = serializedObject.FindProperty("target3DMesh");
		primitiveCount = serializedObject.FindProperty("primitiveCount");
		primitiveInitSize = serializedObject.FindProperty("primitiveInitSize");
		primitiveInitSeed = serializedObject.FindProperty("primitiveInitSeed");
		initPrimitivesOnMeshSurface = serializedObject.FindProperty("initPrimitivesOnMeshSurface");
		randomViewZoomRange = serializedObject.FindProperty("randomViewZoomRange");

		reset = serializedObject.FindProperty("reset");
		pause = serializedObject.FindProperty("pause");
		stepForward = serializedObject.FindProperty("stepForward");
		displayMode = serializedObject.FindProperty("displayMode");
		separateFreeViewCamera = serializedObject.FindProperty("separateFreeViewCamera");

		optimizer = serializedObject.FindProperty("optimizer");
		lossMode = serializedObject.FindProperty("lossMode");
		doAlphaLoss = serializedObject.FindProperty("doAlphaLoss");
		viewsPerOptimStep = serializedObject.FindProperty("viewsPerOptimStep");
		optimizeColorsSeparately = serializedObject.FindProperty("optimizeColorsSeparately");

		beta1 = serializedObject.FindProperty("beta1");
		beta2 = serializedObject.FindProperty("beta2");
		learningRatePosition = serializedObject.FindProperty("learningRatePosition");
		learningRateColor = serializedObject.FindProperty("learningRateColor");

		doPrimitiveResampling = serializedObject.FindProperty("doPrimitiveResampling");
		resamplingInterval = serializedObject.FindProperty("resamplingInterval");
		optimStepsUnseenBeforeKill = serializedObject.FindProperty("optimStepsUnseenBeforeKill");
		minPrimitiveWorldArea = serializedObject.FindProperty("minPrimitiveWorldArea");

		totalElapsedSeconds = serializedObject.FindProperty("totalElapsedSeconds");
		millisecondsPerOptimStep = serializedObject.FindProperty("millisecondsPerOptimStep");
	}

	public void ValidateParameters()
	{
		// Validate parameters
		if (doPrimitiveResampling.boolValue == true)
			primitiveCount.intValue = math.max(primitiveCount.intValue, 2);
		viewsPerOptimStep.intValue = math.max(viewsPerOptimStep.intValue, 1);
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
		EditorGUI.EndDisabledGroup();
		EditorGUILayout.Space();

		// Scene settings
		EditorGUILayout.LabelField("Scene Settings", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(targetResolution);
		EditorGUILayout.PropertyField(target3DMesh);
		EditorGUILayout.PropertyField(primitiveInitSeed);
		EditorGUILayout.PropertyField(primitiveCount);
		EditorGUILayout.PropertyField(primitiveInitSize);
		EditorGUILayout.PropertyField(initPrimitivesOnMeshSurface);
		EditorGUILayout.PropertyField(randomViewZoomRange);

		// Controls
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Controls", EditorStyles.boldLabel);
		if (pause.boolValue == true)
			if (GUILayout.Button("Step Forward"))
				stepForward.boolValue = true;
		EditorGUILayout.PropertyField(pause);
		EditorGUILayout.PropertyField(displayMode);
		EditorGUILayout.PropertyField(separateFreeViewCamera);

		// Optimizer settings
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Optimizer Settings", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(optimizer);
		EditorGUILayout.PropertyField(lossMode);
		EditorGUILayout.PropertyField(doAlphaLoss);
		EditorGUILayout.PropertyField(viewsPerOptimStep);
		EditorGUILayout.PropertyField(optimizeColorsSeparately);

		// Optimizer controls
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Optimizer Controls", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(beta1);
		EditorGUILayout.PropertyField(beta2);
		EditorGUILayout.PropertyField(learningRatePosition);
		EditorGUILayout.PropertyField(learningRateColor);

		// Resampling settings
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Resampling Settings", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(doPrimitiveResampling);
		EditorGUILayout.PropertyField(resamplingInterval);
		EditorGUILayout.PropertyField(optimStepsUnseenBeforeKill);
		EditorGUILayout.PropertyField(minPrimitiveWorldArea);

		ValidateParameters();

		serializedObject.ApplyModifiedProperties();
	}
}
