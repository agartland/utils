import subprocess
import pandas as pd
import tempfile
import os

__all__ = ['runRscript']

def runRscript(Rcmd, inDf=None, outputFiles=0, removeTempFiles=None, Rpath=None):
    """Runs an R cmd with option to provide a DataFrame as input and file
    as output.

    Params
    ------
    Rcmd : str
        String containing the R-script to run.
    inDf : pd.DataFrame or list of pd.DataFrame's
        Data to be passed to the R script via a CSV file.
        Object should be referenced in the script as "INPUTDF" or "INPUTDF0" etc. if list
    outputFiles : int
        Number of output CSV files available for writing by the R-script.
        The contents of the file are returned as a pd.DataFrame.
        File name should be referenced as "OUTPUTFNX" in the R-script
    removeTempFiles : True, False or None
        For debugging. If True then the temporary script and data files will
        always be removed. If None then they will be removed if there is not an error.
        If False they will not be removed.
    Rpath : str
        Optionally provide path to Rscript if not on PATH

    Returns
    -------
    stdout : str
        Output of the R-script at the terminal (including stderr)
    output : pd.DataFrame or list of pd.DataFrames
        Optionally, the contents of CSV file(s) written by the R-script as a pd.DataFrame"""

    """Write data to a tempfile if required"""
    if not inDf is None:
        if not type(inDf) is list:
            inputH, inputFn = tempfile.mkstemp(suffix='.csv', prefix='tmp-Rinput-', text=True)
            readCmd = 'INPUTDF <- read.csv("%s")\n' % inputFn
            Rcmd = readCmd + Rcmd
            os.close(inputH)
            inDf.to_csv(inputFn)
            inputFilenames = [inputFn]
        else:
            inputFilenames = []
            for i, idf in enumerate(inDf):
                inputH, inputFn = tempfile.mkstemp(suffix='.csv', prefix='tmp-Rinput%d-' % i, text=True)
                readCmd = 'INPUTDF%d <- read.csv("%s")\n' % (i, inputFn)
                Rcmd = readCmd + Rcmd
                os.close(inputH)
                idf.to_csv(inputFn)
                inputFilenames.append(inputFn)
                
    """Set up an output file if required"""
    outFn = []
    for outi in range(outputFiles):
        outputH, outputFn = tempfile.mkstemp(suffix='.txt', prefix='tmp-Routput-', text=True)
        outCmd = 'OUTPUTFN%d <- "%s"\n' % (outi, outputFn)
        Rcmd = outCmd + Rcmd
        outFn.append(outputFn)
        os.close(outputH)
        
    """Write script to tempfile"""
    scriptH, scriptFn = tempfile.mkstemp(suffix='.R', prefix='tmp-Rscript-', text=True)
    with open(scriptFn, 'w') as fh:
        fh.write(Rcmd)
    os.close(scriptH)

    """Run the R script and collect output"""
    try:
        if Rpath is None:
            cmdList = ['Rscript', '--vanilla', scriptFn]
        else:
            cmdList = [Rpath, '--vanilla', scriptFn]
        res = subprocess.check_output(cmdList, stderr=subprocess.STDOUT)
        try:
            res = res.decode('utf-8')
        except AttributeError:
            pass
    except subprocess.CalledProcessError as e:
        print('R process returned an error')
        try:
            res = 'STDOUT:\n%s\n' % (e.stdout.decode('utf-8'))
        except AttributeError:
            res = 'STDOUT:\nNone\n'
        print(res)
        try:
            res = 'STDERR:\n%s\n' % (e.stderr.decode('utf-8'))
        except AttributeError:
            res = 'STDERR:\nNone\n'
        print(res)

        if removeTempFiles is None:
            print('Leaving tempfiles for debugging.')
            print(' '.join(cmdList))
            if not inDf is None:
                print(inputFilenames)
            for outputFn in outFn:
                print(outputFn)
            removeTempFiles = False

    """Read the ouptfile if required"""
    outDf = []
    for outputFn in outFn:
        try:
            tmp = pd.read_csv(outputFn)
            outDf.append(tmp)
        except:
            print('Cannot read output CSV: reading as text (%s)' % outputFn)
            with open(outputFn, 'r') as fh:
                tmp = fh.read()
                if len(tmp) == 0:
                    print('Output file is empty! (%s)' % outputFn)
                    tmp = None
                outDf.append(tmp)
    # outDf = [pd.read_csv(outputFn) for outputFn in outFn]
    if len(outDf) == 0:
        outDf = None
    elif len(outDf) == 1:
        outDf = outDf[0]

    """Cleanup the temporary files"""
    if removeTempFiles is None or removeTempFiles:
        os.remove(scriptFn)

        if not inDf is None:
            if not type(inDf) is list:
                os.remove(inputFn)
            else:
                for inputFn in inputFilenames:
                    os.remove(inputFn)
    else:
        print('Leaving tempfiles for debugging.')
        print(' '.join(cmdList))
        if not inDf is None:
            print(inputFn)
        for outputFn in outFn:
            print(outputFn)

    if outputFiles == 0:
        return res
    else:
        return res, outDf

def _test_simple():
    Rcmd = """ctl <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
trt <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
group <- gl(2, 10, 20, labels = c("Ctl","Trt"))
weight <- c(ctl, trt)
lm.D9 <- lm(weight ~ group)
lm.D90 <- lm(weight ~ group - 1) # omitting intercept
anova(lm.D9)
summary(lm.D90)"""
    res = runRscript(Rcmd)
    print(res)

def _test_io():
    ctrl = [4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14]
    trt =  [4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69]
    inDf = pd.DataFrame({'weight':ctrl + trt,
                         'group': ['Ctl']*len(ctrl) + ['Trt']*len(trt)})

    Rcmd = """print(head(INPUTDF))
lm.D9 <- lm(weight ~ group, data=INPUTDF)
lm.D90 <- lm(weight ~ group - 1, data=INPUTDF) # omitting intercept
anova(lm.D9)
summary(lm.D90)
write.csv(data.frame(summary(lm.D90)$coefficients), OUTPUTFN)
"""
    
    res, outputFile = runRscript(Rcmd, inDf=inDf, outputFiles=1)
    print(res)
    print(outputFile)
