# Record the scripted generator sweep (--demo-sweep) as a real-time screen capture of the app window.
#   .\scripts\record_generator_demo.ps1 -Level 8 -Out docs\generator\demo_live_l8.mp4
# Needs ffmpeg on PATH (or pass -Ffmpeg). The app closes itself when the sweep ends.
# The window is a GPU swapchain, so it is captured from the composited desktop and cropped to the window's bounds.
param(
    [int]$Level = 8,
    [string]$Out = "docs\generator\demo_live.mp4",
    [string]$Ffmpeg = "ffmpeg",
    [string]$Size = "1280x1000",
    [int]$Seconds = 46
)
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Win {
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr h, int attr, out RECT r, int size);
}
"@
[void][Win]::SetProcessDPIAware()
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "target\release\sculpt-app.exe"
$outPath = Join-Path $root $Out
$app = Start-Process $exe -ArgumentList @("--level", $Level, "--size", $Size, "--zoom", "0.7", "--demo-sweep") -PassThru
$r = New-Object Win+RECT
$h = [IntPtr]::Zero
# winit creates a tiny helper window first; wait for the real one to reach a sensible size.
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Milliseconds 100
    $app.Refresh()
    $h = $app.MainWindowHandle
    if ($h -ne [IntPtr]::Zero) {
        [void][Win]::DwmGetWindowAttribute($h, 9, [ref]$r, [System.Runtime.InteropServices.Marshal]::SizeOf($r))
        if (($r.Right - $r.Left) -ge 600 -and ($r.Bottom - $r.Top) -ge 400) { break }
    }
}
if (($r.Right - $r.Left) -lt 600) { Write-Error "the app window never reached a usable size"; exit 1 }
[void][Win]::SetForegroundWindow($h)
Start-Sleep -Milliseconds 800
[void][Win]::DwmGetWindowAttribute($h, 9, [ref]$r, [System.Runtime.InteropServices.Marshal]::SizeOf($r))
$w = [int]([math]::Floor(($r.Right - $r.Left) / 2) * 2)
$ht = [int]([math]::Floor(($r.Bottom - $r.Top) / 2) * 2)
Write-Host "recording window at $($r.Left),$($r.Top) size ${w}x${ht}"
& $Ffmpeg -y -f gdigrab -framerate 30 -draw_mouse 0 -offset_x $r.Left -offset_y $r.Top -video_size "${w}x${ht}" -i desktop -t $Seconds `
    -vf "format=yuv420p" -c:v libx264 -preset veryfast -crf 20 $outPath 2>&1 | Select-Object -Last 3
if (-not $app.HasExited) { $app.WaitForExit(15000) | Out-Null }
if (-not $app.HasExited) { $app.Kill() }
Write-Host "wrote $outPath"


