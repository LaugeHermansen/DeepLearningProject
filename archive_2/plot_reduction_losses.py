# %%

temp_reductions =  np.round(np.linspace(0.3, 1, 8), 2)

losses_img_mean = np.array([[np.mean(losses[r_freq,r_time]) for r_time in temp_reductions]  for r_freq in reversed(temp_reductions)])
losses_img_sd = np.array([[np.std(losses[r_freq,r_time], ddof=1)*1.96/len(indices)**0.5 for r_time in temp_reductions]  for r_freq in reversed(temp_reductions)])/losses_img_mean


fig, axs = plt.subplots(1,2, figsize=(8,3.5), dpi=400)

for i, ax, array, title in zip(range(2), axs, (losses_img_mean,losses_img_sd), ("Mean error", "$95$% confidence interval width")):
    im = ax.imshow(array,cmap=get_cmap())
    ax.set_xticks(np.arange(len(temp_reductions)))
    ax.set_yticks(np.arange(len(temp_reductions)))
    ax.set_xticklabels(temp_reductions)
    ax.set_yticklabels(reversed(temp_reductions))

    ax.set_ylabel("Scale on frequency axis")
    ax.set_xlabel("Scale on time axis")
    plt.colorbar(im,ax=ax)
    for pos in ['right', 'top', 'bottom', 'left']:
        ax.spines[pos].set_visible(False)
    ax.set_title(title)

fig.suptitle("Generation quality when reducing quality of spectrogram")#, fontsize=18)
fig.tight_layout()

plt.savefig("poster_report/reduced_generation_quality", bbox_inches = 'tight')

#%%