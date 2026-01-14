using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class DisplayRenderTexture : MonoBehaviour
{
	public Texture displayRenderTexture;

	// Start is called before the first frame update
	void Start()
	{

	}

	// Update is called once per frame
	void Update()
	{

	}

	void OnRenderImage(RenderTexture source, RenderTexture destination)
	{
		if (displayRenderTexture != null)
			Graphics.Blit(displayRenderTexture, destination);
		else
			Graphics.Blit(source, destination);
	}
}
