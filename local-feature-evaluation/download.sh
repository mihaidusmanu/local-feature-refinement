# Create data directory.
mkdir /data/LFE

# cd into data directory.
cd /data/LFE

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

# Download HPatches datasets
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar xvzf hpatches-sequences-release.tar.gz

# Remove the high-resolution sequences
cd hpatches-sequences-release
rm -rf i_contruction i_crownnight i_dc i_pencils i_whitebuilding v_artisans v_astronautis v_talent
cd ..

# wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-release.tar.gz
# tar xvzf hpatches-release.tar.gz
# # Remove the high-resolution sequences
# cd hpatches-release
# rm -rf i_contruction i_crownnight i_dc i_pencils i_whitebuilding v_artisans v_astronautis v_talent
# cd ..


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
cd /home/devs/local-feature-refinement

# create soft link
ln -s /data/LFE /home/devs/local-feature-refinement/LFE
