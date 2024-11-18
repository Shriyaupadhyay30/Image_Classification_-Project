g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize= (10, 10))

#Picking plot style
plt.style.use('bmh')

#Colour theme
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.coolwarm)
plt.title('Confusion Matrix', fontsize=15)
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation= 45, fontsize=10)
plt.yticks(tick_marks, classes, fontsize=10)


thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')
plt.tight_layout()
plt.ylabel('True Label', fontsize=15)
plt.xlabel('Predicted Label', fontsize=15)

plt.show()