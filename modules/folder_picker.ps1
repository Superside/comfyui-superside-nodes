# Shows the normal Windows Explorer folder dialog (IFileOpenDialog with FOS_PICKFOLDERS) and prints the chosen path.
# The older FolderBrowserDialog of Windows PowerShell 5.1 opens a small tree window instead, which is why this
# talks to the shell dialog directly. A folder dialog never lists files, so there are no thumbnails to browse by;
# -PickFile opens the ordinary Open dialog instead (files and thumbnails visible) and the caller takes the folder
# of whatever file was picked. Usage:  powershell -NoProfile -STA -File folder_picker.ps1 [-PickFile] [-NoShow]
param([switch]$NoShow, [switch]$PickFile)

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class SupersideFolderPicker
{
    [ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
    private class FileOpenDialogCom { }

    [ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IFileOpenDialog
    {
        [PreserveSig] int Show(IntPtr parent);
        void SetFileTypes(uint count, IntPtr types);
        void SetFileTypeIndex(uint index);
        void GetFileTypeIndex(out uint index);
        void Advise(IntPtr events, out uint cookie);
        void Unadvise(uint cookie);
        void SetOptions(uint options);
        void GetOptions(out uint options);
        void SetDefaultFolder(IShellItem item);
        void SetFolder(IShellItem item);
        void GetFolder(out IShellItem item);
        void GetCurrentSelection(out IShellItem item);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
        void GetResult(out IShellItem item);
    }

    [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IShellItem
    {
        void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
        void GetParent(out IShellItem parent);
        void GetDisplayName(uint sigdn, [MarshalAs(UnmanagedType.LPWStr)] out string name);
    }

    [DllImport("user32.dll")] private static extern IntPtr GetForegroundWindow();

    public static bool CanCreate()
    {
        var dialog = (IFileOpenDialog)new FileOpenDialogCom();
        uint options; dialog.GetOptions(out options);
        Marshal.ReleaseComObject(dialog);
        return true;
    }

    public static string Pick(string title, bool folder)
    {
        var dialog = (IFileOpenDialog)new FileOpenDialogCom();
        try
        {
            uint options; dialog.GetOptions(out options);
            dialog.SetOptions(options | 0x40 | (folder ? 0x20u : 0u));   // FOS_FORCEFILESYSTEM, + FOS_PICKFOLDERS for a folder
            dialog.SetTitle(title);
            if (dialog.Show(GetForegroundWindow()) != 0) return "";   // cancelled
            IShellItem item; dialog.GetResult(out item);
            string path; item.GetDisplayName(0x80058000, out path);   // SIGDN_FILESYSPATH
            return path ?? "";
        }
        finally { Marshal.ReleaseComObject(dialog); }
    }
}
"@

if ($NoShow) { [Console]::Out.Write([SupersideFolderPicker]::CanCreate()) }
elseif ($PickFile) { [Console]::Out.Write([SupersideFolderPicker]::Pick("Open the product folder and pick any file in it (an image or product.json)", $false)) }
else { [Console]::Out.Write([SupersideFolderPicker]::Pick("Choose the folder where the product will be saved", $true)) }
