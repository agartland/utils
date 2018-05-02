from glob import glob
from quickr import runRscript

__all__ = ['convertGatingSet']

_RCMD = """
library(feather)
library(flowWorkspace)
library(data.table)

#folder <- 'X:/fast/gilbert_p/grp/compass_hvtn602_aw/tmpdata/774-2'
gs <- flowWorkspace::load_gs(folder)

convertGH = function(gh){
    nodes <- getNodes(gh)[-1]
    #init data table with first column
    first.node <- nodes[1]
    ind.list <- list(getIndices(gh, first.node))
    names(ind.list) <- first.node
    dt <- data.table(as.data.frame(ind.list, check.names = F))

    #add rest columns iteratively 
    for (n in nodes[-1]){
        ind <- getIndices(gh, n)
        dt[, eval(n) := ind]
    }
    #dt <- sapply(dt, as.integer)
    return(as.data.frame(dt))
    }

for (i in 1:length(gs)){
    df <- convertGH(gs[[i]])
    write_feather(df, paste0(folder, '/gs_', toString(i), '_sample_', sampleNames(gs[[i]]), '.feather'))
    print(paste0('gs_', toString(i), '_sample_', sampleNames(gs[[i]]), '.feather'))
}

write.csv(pData(gs), file='metadata.csv')
"""

def convertGatingSet(folder):
    RCMD = 'folder <- "%s"\n%s' % (folder, _RCMD)
    res = runRscript(RCMD)

if __name__ == '__main__':
    import sys
    convertGatingSet(sys.argv[1])

