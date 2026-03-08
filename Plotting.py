
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler




# قراءة الملف مع تجاهل التعليقات
df = pd.read_csv("GSE65904_series_matrix.txt", sep="\t", comment="!")

# تعيين العمود النصي كـ index (غالبًا اسمه ID_REF)
df.set_index("ID_REF", inplace=True)

# اختيار الأعمدة الرقمية فقط
numeric_df = df.select_dtypes(include=["number"])
# حساب الانحراف المعياري لكل جين واختيار الأعلى
gene_variance = numeric_df.var(axis=1)
top_genes = numeric_df.loc[gene_variance.sort_values(ascending=False).head(500).index]

# حساب correlation بين الجينات المختارة
corr_genes = top_genes.T.corr()

# رسم heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr_genes, cmap="coolwarm")
plt.title("Correlation Heatmap of Top 500 Variable Genes")
plt.savefig("Correlation Heatmap of Melanoma Sample Clustering.png")
plt.show()



# اختيار أعلى 20 جين متغير
top20_genes = gene_variance.sort_values(ascending=False).head(20)

# رسم barplot
plt.figure(figsize=(12,6))
sns.barplot(x=top20_genes.index, y=top20_genes.values, palette="viridis")
plt.xticks(rotation=90)
plt.title("Top 20 Variable Genes")
plt.ylabel("Variance")
plt.xlabel("Gene ID")
plt.savefig("Top 20 Variable Gene.png")
plt.show()

















# اختيار أول 3 جينات من حيث التباين
top3_genes = top20_genes.index[:3]

# Barplot لعرض متوسط التعبير لكل جين
mean_expression = numeric_df.loc[top3_genes].mean(axis=1)

plt.figure(figsize=(8,6))
sns.barplot(x=mean_expression.index, y=mean_expression.values, palette="mako")
plt.title("Mean Expression of Top 3 Variable Genes")
plt.ylabel("Mean Expression")
plt.xlabel("Gene ID")
plt.savefig("Top 3 Variable Genes (barplot).png")
plt.show()


# Boxplot لعرض توزيع التعبير لكل جين
plt.figure(figsize=(10,6))
sns.boxplot(data=numeric_df.loc[top3_genes].T)
plt.title("Boxplot of Top 3 Variable Genes")
plt.xlabel("Gene ID")
plt.ylabel("Expression Level")


plt.figure(figsize=(10,6))
sns.violinplot(data=numeric_df.loc[top3_genes].T)
plt.title("violinplot of Top 3 Variable Genes")
plt.xlabel("Gene ID")
plt.ylabel("Expression Level")

plt.tight_layout()
plt.savefig("Top 3 Variable Genes (violinplot).png")
plt.show()


















# قراءة ملف الـ annotation
anno = pd.read_csv("GPL10558-50081.txt", sep="\t", comment="#", low_memory=False)  # غيّر الاسم حسب الملف
# افترض أن العمود فيه probe ID اسمه 'ID' والجين اسمه 'Gene Symbol'
print(anno.columns)

merged = df.merge(anno[['ID','Symbol']], left_index=True, right_on='ID')
merged.set_index('Symbol', inplace=True)


# قائمة الجينات المناعية
immune_genes = ["CD8A", "CD4", "FOXP3", "CD68", "PDCD1"]# فلترة الداتا للجينات المطلوبة
immune_df = merged.loc[merged.index.intersection(immune_genes)]



immune_numeric = immune_df.select_dtypes(include=["number"])# Boxplot لعرض توزيع التعبير لكل جين
plt.figure(figsize=(10,6))
sns.boxplot(data=immune_numeric.T)
plt.title("Boxplot of Immune Genes Expression")
plt.xlabel("Gene")
plt.ylabel("Expression Level")
plt.show()




# اختيار الأعمدة الرقمية فقط من immune_df
immune_numeric = immune_df.select_dtypes(include=["number"])# حساب المتوسط لكل جين
mean_expression = immune_numeric.mean(axis=1)# Barplot لعرض متوسط التعبير لكل جين
plt.figure(figsize=(8,6))
sns.barplot(x=mean_expression.index, y=mean_expression.values, palette="mako")
plt.title("Mean Expression of Immune Genes")
plt.ylabel("Mean Expression")
plt.xlabel("Gene")
plt.show()








# قائمة الجينات المناعية
proliferation_genes = ["MKI67","PCNA", "CCNB1","CDK1"]# فلترة الداتا للجينات المطلوبة
proliferation_df = merged.loc[merged.index.intersection(proliferation_genes)]



proliferation_numeric = proliferation_df.select_dtypes(include=["number"])# Boxplot لعرض توزيع التعبير لكل جين
plt.figure(figsize=(10,6))
sns.boxplot(data=proliferation_numeric.T)
plt.title("Boxplot of tumor proliferation genes")
plt.xlabel("Gene")
plt.ylabel("Expression Level")
plt.show()



# اختيار الأعمدة الرقمية فقط من immune_df
proliferation_numeric = proliferation_df.select_dtypes(include=["number"])# حساب المتوسط لكل جين
mean_expression = proliferation_numeric.mean(axis=1)# Barplot لعرض متوسط التعبير لكل جين
plt.figure(figsize=(8,6))
sns.barplot(x=mean_expression.index, y=mean_expression.values, palette="mako")
plt.title("Mean tumor proliferation genes")
plt.ylabel("Mean Expression")
plt.xlabel("Gene")
plt.show()









proteins = ["MKI67","PCNA", "CCNB1","CDK1","CD8A", "CD4", "FOXP3", "CD68", "PDCD1"]

# فلترة البيانات من merged
subset = merged.loc[merged.index.intersection(proteins)].select_dtypes(include=["number"]).T.dropna()

# توحيد القيم (Scaling)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(subset)

# --- PCA ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Projection of Protein Expression")
plt.savefig("pca_projection.png")
plt.show()








