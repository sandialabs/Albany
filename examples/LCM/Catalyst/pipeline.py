
try: paraview.simple
except: from paraview.simple import *

from paraview import coprocessing


#--------------------------------------------------------------
# Code generated from cpstate.py to create the CoProcessor.


# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      RenderView1 = coprocessor.CreateView( CreateRenderView, "image_%t.png", 1, 0, 1, 1188, 850 )
      RenderView1.CacheKey = 0.0
      RenderView1.Background = [0.31999694819562063, 0.3400015259021897, 0.4299992370489052]
      RenderView1.UseLight = 1
      RenderView1.CameraPosition = [4.999999999999999, 0.5000000000000007, 19.414868826867117]
      RenderView1.LightSwitch = 0
      RenderView1.CameraClippingRange = [19.220720138598445, 19.706091859270124]
      RenderView1.ViewTime = 0.0
      RenderView1.InteractionMode = '2D'
      RenderView1.CameraFocalPoint = [4.999999999999999, 0.5000000000000007, 0.0]
      RenderView1.CameraParallelScale = 5.024937810560444
      RenderView1.CenterOfRotation = [4.999999999999999, 0.5000000000000007, 0.0]
      
      filename_0_pvtu = coprocessor.CreateProducer( datadescription, "input" )
      
      a1_Scalars__PiecewiseFunction = CreatePiecewiseFunction( Points=[0.0, 0.0, 0.5, 0.0, 0.9999999999999999, 1.0, 0.5, 0.0] )
      
      a1_Scalars__PVLookupTable = GetLookupTableForArray( "Scalars_", 1, RGBPoints=[0.0, 0.23, 0.299, 0.754, 0.9999999999999999, 0.706, 0.016, 0.15], VectorMode='Magnitude', NanColor=[0.25, 0.0, 0.0], ScalarOpacityFunction=a1_Scalars__PiecewiseFunction, ColorSpace='Diverging', ScalarRangeInitialized=1.0 )
      
      ScalarBarWidgetRepresentation1 = CreateScalarBar( TitleFontSize=12, Title='Scalars_', Enabled=1, LookupTable=a1_Scalars__PVLookupTable, LabelFontSize=12 )
      GetRenderView().Representations.append(ScalarBarWidgetRepresentation1)
      
      DataRepresentation1 = Show()
      DataRepresentation1.EdgeColor = [0.0, 0.0, 0.5000076295109483]
      DataRepresentation1.SelectionPointFieldDataArrayName = 'Scalars_'
      DataRepresentation1.ScalarOpacityFunction = a1_Scalars__PiecewiseFunction
      DataRepresentation1.ColorArrayName = ('POINT_DATA', 'Scalars_')
      DataRepresentation1.ScalarOpacityUnitDistance = 0.6557322144483001
      DataRepresentation1.LookupTable = a1_Scalars__PVLookupTable
      DataRepresentation1.ScaleFactor = 0.9999999999999999
      
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  freqs = {'input': [1]}
  coprocessor.SetUpdateFrequencies(freqs)
  return coprocessor

#--------------------------------------------------------------
# Global variables that will hold the pipeline for each timestep
# Creating the CoProcessor object, doesn't actually create the ParaView pipeline.
# It will be automatically setup when coprocessor.UpdateProducers() is called the
# first time.
coprocessor = CreateCoProcessor()

#--------------------------------------------------------------
# Enable Live-Visualizaton with ParaView
coprocessor.EnableLiveVisualization(False)


# ---------------------- Data Selection method ----------------------

def RequestDataDescription(datadescription):
    "Callback to populate the request for current timestep"
    global coprocessor
    if datadescription.GetForceOutput() == True:
        # We are just going to request all fields and meshes from the simulation
        # code/adaptor.
        for i in range(datadescription.GetNumberOfInputDescriptions()):
            datadescription.GetInputDescription(i).AllFieldsOn()
            datadescription.GetInputDescription(i).GenerateMeshOn()
        return

    # setup requests for all inputs based on the requirements of the
    # pipeline.
    coprocessor.LoadRequestedData(datadescription)

# ------------------------ Processing method ------------------------

def DoCoProcessing(datadescription):
    "Callback to do co-processing for current timestep"
    global coprocessor

    # Update the coprocessor by providing it the newly generated simulation data.
    # If the pipeline hasn't been setup yet, this will setup the pipeline.
    coprocessor.UpdateProducers(datadescription)

    # Write output data, if appropriate.
    coprocessor.WriteData(datadescription);

    # Write image capture (Last arg: rescale lookup table), if appropriate.
    coprocessor.WriteImages(datadescription, rescale_lookuptable=False)

    # Live Visualization, if enabled.
    coprocessor.DoLiveVisualization(datadescription, "localhost", 22222)
