using System;
using System.IO;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

[Serializable]
public class SceneInfo
{
    public List<ObjInfo> objs = new List<ObjInfo>();
}

[Serializable]
public class ObjInfo
{
    public string id;
    public string modelId;
    public int index;
    public string parentId;
    public int parentIndex;
    public ObjTransform transform;
}

[Serializable]
public class ObjTransform
{
    public int rows;
    public int cols;
    public float[] data;
}

[InitializeOnLoad]
public static class PythonUnityBridge
{
    private static readonly string CommandPath = "Temp/bridgeio_command.txt";
    private static readonly string StatePath = "Temp/bridgeio_state.txt";
    private static readonly string OutputPath = "Temp/bridgeio_results.json";
    private static FileSystemWatcher watcher;
    private static string pendingCommand = null;

    static PythonUnityBridge()
    {
        // Monitor the command file for change, indicating a new command from Python side
        watcher = new FileSystemWatcher(Path.GetDirectoryName(CommandPath), Path.GetFileName(CommandPath));
        watcher.NotifyFilter = NotifyFilters.LastWrite;
        watcher.Changed += OnCommandFileChanged;
        watcher.EnableRaisingEvents = true;

        // Update state file on play mode changes and process commands in the update loop
        EditorApplication.playModeStateChanged += _ => UpdateStateFile(GetPlayModeState());
        EditorApplication.update += ProcessPendingCommand;

        // Initialize state file
        UpdateStateFile(GetPlayModeState());
    }

    private static string GetPlayModeState()
    {
        return EditorApplication.isPlaying ? "PLAYING" : "STOPPED";
    }

    private static void UpdateStateFile(string state)
    {
        try
        {
            File.WriteAllText(StatePath, state);
            Debug.Log($"[SceneEvalConverter] > Updated state file: {state}");
        }
        catch (IOException ex)
        {
            Debug.LogError($"[SceneEvalConverter] > Error writing state file: {ex.Message}");
        }
    }

    private static void OnCommandFileChanged(object source, FileSystemEventArgs e)
    {
        try
        {
            pendingCommand = File.ReadAllText(CommandPath).Trim();
        }
        catch (IOException ex)
        {
            Debug.LogError($"[SceneEvalConverter] > Error reading command file: {ex.Message}");
        }
    }

    private static void ProcessPendingCommand()
    {
        if (!string.IsNullOrEmpty(pendingCommand))
        {
            Debug.Log($"[SceneEvalConverter] > Command from Python detected: {pendingCommand}");
            ProcessCommand(pendingCommand);
            pendingCommand = null;
        }
    }

    private static void ProcessCommand(string command)
    {
        Debug.Log($"[SceneEvalConverter] > Processing command: {command}");

        if (command == "PLAY" && !EditorApplication.isPlaying)
        {
            Debug.Log($"[SceneEvalConverter] > Entering play mode");
            EditorApplication.isPlaying = true;
        }
        else if (command == "STOP" && EditorApplication.isPlaying)
        {
            Debug.Log($"[SceneEvalConverter] > Exiting play mode");
            EditorApplication.isPlaying = false;
        }
        else if (command == "PROCESS")
        {
            Debug.Log($"[SceneEvalConverter] > Processing...");
            string response = ExtractInfo();
            Debug.Log($"[SceneEvalConverter] > Processing completed");
            try
            {
                File.WriteAllText(OutputPath, response);
                Debug.Log($"[SceneEvalConverter] > Wrote PROCESS response to: {OutputPath}");
                UpdateStateFile("PROCESSED");
            }
            catch (IOException ex)
            {
                Debug.LogError($"[SceneEvalConverter] > Error writing PROCESS response: {ex.Message}");
            }
        }
    }

    private static string ExtractInfo()
    {
        SceneInfo sceneInfo = new SceneInfo();

        // "Objects" hold all loaded objects in the scene
        GameObject ObjectRoot = GameObject.Find("Objects");
        if (ObjectRoot != null)
        {
            int index = 0;
            foreach (Transform child in ObjectRoot.transform)
            {
                // Skip doors and windows objects
                if (child.name.StartsWith("door") || child.name.StartsWith("window"))
                    continue;

                SimObjPhysics metadata = child.GetComponent<SimObjPhysics>();

                // Skip Thor objects
                if (metadata.assetID.Length < "ac92f6e97eaa43c4ad6cb8f7c65ac43f".Length)
                    continue;

                // Convert from Unity to SceneEval coordinate system
                Vector3 b_pos = new Vector3(child.position.x, child.position.z, child.position.y);
                Quaternion b_rot = Quaternion.Euler(90, 0, 0) * Quaternion.Euler(child.eulerAngles.x, -(child.eulerAngles.y - 180f), -child.eulerAngles.z);
                Vector3 b_scale = new Vector3(child.localScale.x, child.localScale.z, child.localScale.y);
                Matrix4x4 transformed_matrix = Matrix4x4.TRS(b_pos, b_rot, b_scale);

                // Create entry
                ObjInfo objInfo = new ObjInfo()
                {
                    id = (index + 1).ToString(),
                    modelId = $"objaverse.{metadata.assetID}",
                    index = index,
                    parentId = "-1",
                    parentIndex = -1,
                    transform = new ObjTransform()
                    {
                        rows = 4,
                        cols = 4,
                        data = new float[]
                        {
                            transformed_matrix[0,0], transformed_matrix[1,0], transformed_matrix[2,0], transformed_matrix[3,0],
                            transformed_matrix[0,1], transformed_matrix[1,1], transformed_matrix[2,1], transformed_matrix[3,1],
                            transformed_matrix[0,2], transformed_matrix[1,2], transformed_matrix[2,2], transformed_matrix[3,2],
                            transformed_matrix[0,3], transformed_matrix[1,3], transformed_matrix[2,3], transformed_matrix[3,3],
                        },
                    }
                };

                Debug.Log($"[SceneEvalConverter] > Extracted ObjInfo: id={objInfo.id}, modelId={objInfo.modelId}");

                sceneInfo.objs.Add(objInfo);
                index++;
            }
        }

        Debug.Log($"[SceneEvalConverter] > {sceneInfo.objs.Count} objects extracted.");
        Debug.Log($"[SceneEvalConverter] > Extraction complete.");

        return JsonUtility.ToJson(sceneInfo);
    }

}
