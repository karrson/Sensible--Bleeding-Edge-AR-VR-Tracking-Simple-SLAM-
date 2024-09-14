using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;

public class VRCam : MonoBehaviour
{
    public string PathToExe = "";
    public System.Diagnostics.Process VRCamProcess;

    [DllImport("user32.dll")] static extern uint GetActiveWindow();
    [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr hWnd);

    public void Start()
    {
        foreach (Process p in Process.GetProcessesByName(Path.GetFileNameWithoutExtension(PathToExe))) p.Kill();
        foreach (Process p in Process.GetProcessesByName("RuntimeBroker")) p.Kill();
        VRCamProcess = new Process();
        ProcessStartInfo startInfo = new ProcessStartInfo();
        startInfo.UseShellExecute = false;
        startInfo.CreateNoWindow = true;
        startInfo.FileName = PathToExe;
        startInfo.RedirectStandardOutput = true;
        VRCamProcess.StartInfo = startInfo;
        VRCamProcess.OutputDataReceived += ReceiveMessage;
        VRCamProcess.Start();
        VRCamProcess.BeginOutputReadLine();
    }

    private void ReceiveMessage(object sender, DataReceivedEventArgs e)
    {
        Pose = e.Data
                .Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => float.Parse(s))
                .ToList();
    }

    private List<float> Pose = new List<float>();

    public void LateUpdate()
    {
        if (Pose.Count == 7)
        {
            transform.localPosition = new Vector3(Pose[0], Pose[1], Pose[2]);
            transform.localRotation = new Quaternion(Pose[3], Pose[4], Pose[5], Pose[6]);

            if (VRCamProcess.HasExited) Application.Quit();
        }
    }
}
