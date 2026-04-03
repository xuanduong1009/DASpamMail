# Bo dataset nop rieng cho do an

Thu muc nay duoc tach rieng de nop kem bao cao va source code.

Noi dung gom:

- `source_bilingual/ham`: toan bo email ham goc dang duoc nhom su dung
- `source_bilingual/spam`: toan bo email spam goc dang duoc nhom su dung
- `interim/emails_raw.csv`: du lieu sau buoc parse email
- `processed/train.csv`: tap train sau khi chia du lieu
- `processed/test.csv`: tap test sau khi chia du lieu
- `stopwords_vi.txt`: danh sach stopword tieng Viet phuc vu tien xu ly
- `manifest.csv`: bang liet ke tung file va nhan cua no

Thong ke hien tai:

- So file nguon: 5707
- Ham: 3957
- Spam: 1750
- Mau dung duoc sau parse: 5691
- Train: 4552
- Test: 1139

Ghi chu:

- Phan nen du lieu tieng Anh ke thua tu Enron/Enron Spam corpus mo rong.
- Phan du lieu tieng Viet do nhom tu bo sung de phu hop bai toan loc spam song ngu.
