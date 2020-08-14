# Create data directory.
mkdir LFE

# cd into data directory.
cd LFE

# Download Strecha datasets.
wget http://cvg.ethz.ch/research/local-feature-evaluation/Strecha-Fountain.zip
unzip Strecha-Fountain.zip
rm Strecha-Fountain.zip

wget http://cvg.ethz.ch/research/local-feature-evaluation/Strecha-Herzjesu.zip
unzip Strecha-Herzjesu.zip
rm Strecha-Herzjesu.zip

# Download the internet datasets.
wget http://landmark.cs.cornell.edu/projects/1dsfm/images.Madrid_Metropolis.tar
tar xvf images.Madrid_Metropolis.tar
rm images.Madrid_Metropolis.tar

wget http://landmark.cs.cornell.edu/projects/1dsfm/images.Gendarmenmarkt.tar
tar xvf images.Gendarmenmarkt.tar
mv home/wilsonkl/projects/SfM_Init/dataset_images/Gendarmenmarkt Gendarmenmarkt
rm -r home
rm images.Gendarmenmarkt.tar

wget http://landmark.cs.cornell.edu/projects/1dsfm/images.Tower_of_London.tar
tar xvf images.Tower_of_London.tar
rm images.Tower_of_London.tar

# Download empty databases.
wget http://cvg.ethz.ch/research/local-feature-evaluation/Databases.tar.gz
tar xvzf Databases.tar.gz Fountain/database.db
tar xvzf Databases.tar.gz Herzjesu/database.db
tar xvzf Databases.tar.gz Madrid_Metropolis/database.db
tar xvzf Databases.tar.gz Gendarmenmarkt/database.db
tar xvzf Databases.tar.gz Tower_of_London/database.db
rm Databases.tar.gz

# Download the match-lists.
wget https://dsmn.ml/files/local-feature-refinement/LFE-match-lists.tar
tar xvf LFE-match-lists.tar
rm LFE-match-lists.tar

# cd out of data directory.
cd ..
