$files = Get-ChildItem "./" -Filter *.png -file -Recurse | 
foreach-object {

    $Source = $_.FullName
    $test = [System.IO.Path]::GetDirectoryName($source)
    $base= $_.BaseName+".jpg"
    $basedir = $test+"\"+$base
    Write-Host $basedir
    Add-Type -AssemblyName system.drawing
    $imageFormat = "System.Drawing.Imaging.ImageFormat" -as [type]
    $image = [drawing.image]::FromFile($Source)
    # $image.Save($basedir, $imageFormat::jpeg) Don't save here!

    # Create a new image
    $NewImage = [System.Drawing.Bitmap]::new($Image.Width,$Image.Height)
    $NewImage.SetResolution($Image.HorizontalResolution,$Image.VerticalResolution)

    # Add graphics based on the new image
    $Graphics = [System.Drawing.Graphics]::FromImage($NewImage)
    $Graphics.Clear([System.Drawing.Color]::White) # Set the color to white
    $Graphics.DrawImageUnscaled($image,0,0) # Add the contents of $image

    # Now save the $NewImage instead of $image
    $NewImage.Save($basedir,$imageFormat::Jpeg)

    # Uncomment these two lines if you want to delete the png files:
    $image.Dispose()
    Remove-Item $Source
}  