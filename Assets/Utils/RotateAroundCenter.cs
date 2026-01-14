using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RotateAroundCenter : MonoBehaviour
{
	public float rotateSpeed = 0.0f;

	// Start is called before the first frame update
	void OnEnable()
	{

	}

	// Update is called once per frame
	void Update()
	{
		transform.RotateAround(Vector3.zero, Vector3.up, rotateSpeed);
	}
}
