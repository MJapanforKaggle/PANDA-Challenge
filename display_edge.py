#display edge image with category
i=1
fig_h = 3
fig_w = 5
names = [name[:] for name in os.listdir(TRAIN)]
for name in tqdm(names[:fig_h*fig_w]):
    path = os.path.join(TRAIN, name)
    _pos = name.find("_")
    img = cv2.imread(path,0)
    lap = cv2.Laplacian(img, cv2.CV_32F,ksize=5)
    plt.subplot(fig_h,fig_w,i),plt.imshow(lap, cmap='gray')
    plt.title('Category' + str(df.loc[name[:_pos]].loc['isup_grade'])), plt.xticks([]), plt.yticks([])
    i += 1
plt.show()
