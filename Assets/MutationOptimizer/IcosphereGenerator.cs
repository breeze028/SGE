using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public static class IcosphereGenerator
{
    public static void Generate(
        int subdivision,
        out Vector3[] vertices,
        out int[] triangles)
    {
        float t = (1f + Mathf.Sqrt(5f)) / 2f;

        var vertList = new List<Vector3>
        {
            new Vector3(-1,  t,  0).normalized,
            new Vector3( 1,  t,  0).normalized,
            new Vector3(-1, -t,  0).normalized,
            new Vector3( 1, -t,  0).normalized,

            new Vector3( 0, -1,  t).normalized,
            new Vector3( 0,  1,  t).normalized,
            new Vector3( 0, -1, -t).normalized,
            new Vector3( 0,  1, -t).normalized,

            new Vector3( t,  0, -1).normalized,
            new Vector3( t,  0,  1).normalized,
            new Vector3(-t,  0, -1).normalized,
            new Vector3(-t,  0,  1).normalized,
        };

        var triList = new List<int>
        {
            0,11,5,  0,5,1,  0,1,7,  0,7,10, 0,10,11,
            1,5,9,  5,11,4, 11,10,2, 10,7,6,  7,1,8,
            3,9,4,  3,4,2,  3,2,6,  3,6,8,  3,8,9,
            4,9,5,  2,4,11, 6,2,10, 8,6,7,  9,8,1
        };

        for (int i = 0; i < subdivision; i++)
        {
            var midpointCache = new Dictionary<long, int>();
            var newTris = new List<int>();

            for (int tIdx = 0; tIdx < triList.Count; tIdx += 3)
            {
                int a = triList[tIdx];
                int b = triList[tIdx + 1];
                int c = triList[tIdx + 2];

                int ab = GetMidpoint(a, b, vertList, midpointCache);
                int bc = GetMidpoint(b, c, vertList, midpointCache);
                int ca = GetMidpoint(c, a, vertList, midpointCache);

                newTris.AddRange(new int[]
                {
                    a, ab, ca,
                    b, bc, ab,
                    c, ca, bc,
                    ab, bc, ca
                });
            }

            triList = newTris;
        }

        vertices = vertList.ToArray();
        triangles = triList.ToArray();
    }

    public static void GetAdjacency(
        Vector3[] vertices,
        int[] triangles,
        out int[] adjacency,
        out int[] adjacencyStart,
        out int[] adjacencyCount
    )
    {
        int vertexCount = vertices.Length;
        
        List<List<int>> adjacencyList = new List<List<int>>(vertexCount);
        for (int i = 0; i < vertexCount; i++)
            adjacencyList.Add(new List<int>());

        for (int t = 0; t < triangles.Length; t += 3)
        {
            int v0 = triangles[t + 0];
            int v1 = triangles[t + 1];
            int v2 = triangles[t + 2];

            adjacencyList[v0].Add(v1);
            adjacencyList[v0].Add(v2);

            adjacencyList[v1].Add(v0);
            adjacencyList[v1].Add(v2);

            adjacencyList[v2].Add(v0);
            adjacencyList[v2].Add(v1);
        }
        
        for (int i = 0; i < vertexCount; i++)
            adjacencyList[i] = adjacencyList[i].Distinct().ToList();
        
        int totalAdjacency = 0;
        for (int i = 0; i < vertexCount; i++)
            totalAdjacency += adjacencyList[i].Count;

        adjacency = new int[totalAdjacency];
        adjacencyStart = new int[vertexCount];
        adjacencyCount = new int[vertexCount];

        int offset = 0;
        for (int i = 0; i < vertexCount; i++)
        {
            adjacencyStart[i] = offset;
            int count = adjacencyList[i].Count;
            adjacencyCount[i] = count;

            for (int k = 0; k < count; k++)
                adjacency[offset + k] = adjacencyList[i][k];

            offset += count;
        }
    }
    
    static int GetMidpoint(
        int i0,
        int i1,
        List<Vector3> verts,
        Dictionary<long, int> cache)
    {
        long key = ((long)Mathf.Min(i0, i1) << 32) + Mathf.Max(i0, i1);
        if (cache.TryGetValue(key, out int idx))
            return idx;

        Vector3 mid = ((verts[i0] + verts[i1]) * 0.5f).normalized;
        idx = verts.Count;
        verts.Add(mid);
        cache[key] = idx;
        return idx;
    }
}
