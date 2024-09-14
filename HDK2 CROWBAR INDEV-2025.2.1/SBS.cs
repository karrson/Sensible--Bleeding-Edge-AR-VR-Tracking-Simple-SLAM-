using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SBS : MonoBehaviour
{
    public Material Mat;

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        source.wrapMode = TextureWrapMode.Repeat;
        Graphics.Blit(source, destination, Mat);
    }
}
