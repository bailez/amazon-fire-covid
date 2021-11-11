import matplotlib.pyplot as plt

def plot_seasonalcomp(df, decomp):
    fig, axs = plt.subplots(4,1,figsize=(6,10))
    
    df.plot(ax=axs[0], legend=False)
    axs[0].set(ylabel='log scale', xlabel='')
    
    
    decomp.trend.plot(ax=axs[1])
    axs[1].set(ylabel='trend', xlabel='')
    
    
    decomp.seasonal.plot(ax=axs[2])
    axs[2].set(ylabel='seasonal', xlabel='')
    
    seasonal_resid = decomp.resid.dropna().reset_index()
    axs[3].set_ylim([-1.5,1.5])
    seasonal_resid.plot.scatter(x='Data', y='resid',ax=axs[3])
    axs[3].set(ylabel='resid', xlabel='')
    plt.axhline(0, color='black')
