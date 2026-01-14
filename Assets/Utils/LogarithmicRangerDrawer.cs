using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;

/// <summary>
/// Add this attribute to a float property to make it a logarithmic range slider
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public class LogarithmicRangeAttribute : PropertyAttribute
{
	public float min;
	public float center;
	public float max;

	/// <summary>
	/// Creates a float property slider with a logarithmic 
	/// </summary>
	/// <param name="min">Minimum range value</param>
	/// <param name="center">Value at the center of the range slider</param>
	/// <param name="max">Maximum range value</param>
	public LogarithmicRangeAttribute(float min, float center, float max)
	{
		this.min = min;
		this.center = center;
		this.max = max;
	}
}

#if UNITY_EDITOR
[CustomPropertyDrawer(typeof(LogarithmicRangeAttribute))]
public class LogarithmicRangeDrawer : PropertyDrawer
{
	public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
	{
		LogarithmicRangeAttribute logRangeAttribute = (LogarithmicRangeAttribute)attribute;
		LogRangeConverter rangeConverter = new LogRangeConverter(logRangeAttribute.min, logRangeAttribute.center, logRangeAttribute.max);

		EditorGUI.BeginProperty(position, label, property);

		//GUILayout.BeginHorizontal(GUILayout.Height(10));
		//EditorGUILayout.PrefixLabel(label);
		//float value = rangeConverter.ToNormalized(property.floatValue);
		//value = GUILayout.HorizontalSlider(value, 0, 1);
		//value = rangeConverter.ToRange(value);
		//property.floatValue = float.Parse(value.ToString("G2"));
		//property.floatValue = EditorGUILayout.FloatField(property.floatValue, GUILayout.MaxWidth(60));
		//GUILayout.EndHorizontal();

		float rightWidth = 60.0f; // Max size for the left-most element
		float padding = 5.0f;  // Some space between elements
		position = EditorGUI.PrefixLabel(position, label);
		float centerWidth = position.width - rightWidth - padding * 2;
		Rect centerRect = new Rect(position.x + padding, position.y, centerWidth, position.height);
		Rect rightRect = new Rect(position.x + centerWidth + padding * 2, position.y, rightWidth, position.height);
		float value = rangeConverter.ToNormalized(property.floatValue);
		value = GUI.HorizontalSlider(centerRect, value, 0, 1);
		value = rangeConverter.ToRange(value);
		property.floatValue = float.Parse(value.ToString("G2"));
		property.floatValue = EditorGUI.FloatField(rightRect, property.floatValue);

		EditorGUI.EndProperty();
	}

	public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
	{
		return EditorGUIUtility.singleLineHeight;
	}
}

/// <summary>
/// Tool to convert a range from 0-1 into a logarithmic range with a user defined center
/// </summary>
public struct LogRangeConverter
{
	public readonly float minValue;
	public readonly float maxValue;

	private readonly float a;
	private readonly float b;
	private readonly float c;

	/// <summary>
	/// Set up a scaler
	/// </summary>
	/// <param name="minValue">Value for t = 0</param>
	/// <param name="centerValue">Value for t = 0.5</param>
	/// <param name="maxValue">Value for t = 1.0</param>
	public LogRangeConverter(float minValue, float centerValue, float maxValue)
	{
		this.minValue = minValue;
		this.maxValue = maxValue;

		a = (minValue * maxValue - (centerValue * centerValue)) / (minValue - 2 * centerValue + maxValue);
		b = ((centerValue - minValue) * (centerValue - minValue)) / (minValue - 2 * centerValue + maxValue);
		c = 2 * Mathf.Log((maxValue - centerValue) / (centerValue - minValue));
	}

	/// <summary>
	/// Convers the value in range 0 - 1 to the value in range of minValue - maxValue
	/// </summary>
	public float ToRange(float value01)
	{
		return a + b * Mathf.Exp(c * value01);
	}

	/// <summary>
	/// Converts the value in range min-max to a value between 0 and 1 that can be used for a slider
	/// </summary>
	public float ToNormalized(float rangeValue)
	{
		return Mathf.Log((rangeValue - a) / b) / c;
	}
}
#endif