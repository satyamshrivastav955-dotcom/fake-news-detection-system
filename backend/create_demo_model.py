"""
Fake News Detection - Demo Model Generator
============================================
Creates a compact demo model using a synthetic dataset so the app can run
immediately even without the full ISOT CSV files.

This is ONLY for demonstration purposes. For production accuracy, run
train_model.py with the real ISOT/Kaggle dataset.

Usage:
    python create_demo_model.py
"""

import os
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ─── Synthetic Training Data ──────────────────────────────────────────────────
# Real news characteristics: factual, sourced, measured language
REAL_SAMPLES = [
    "President signed the new infrastructure bill into law on Friday after bipartisan support in Congress.",
    "Scientists at NASA confirmed the discovery of water ice deposits near the lunar south pole.",
    "The Federal Reserve raised interest rates by 25 basis points following the latest inflation report.",
    "A magnitude 6.2 earthquake struck central Italy early Thursday, according to the USGS.",
    "The World Health Organization released updated guidelines on COVID-19 vaccine booster doses.",
    "Tech giant reports quarterly earnings of $2.1 billion, surpassing analyst expectations.",
    "The Supreme Court ruled five to four in favor of expanding voting access in the contested case.",
    "Climate scientists published new data showing accelerated ice melt in Greenland glaciers.",
    "The unemployment rate fell to 3.5 percent in March, according to the Bureau of Labor Statistics.",
    "Prime Minister addressed parliament on the country's new economic recovery plan.",
    "Electric vehicle sales rose 40 percent year over year, reflecting growing consumer demand.",
    "International trade negotiations between the two nations resumed in Geneva on Monday.",
    "The central bank announced a new monetary policy framework to address inflation concerns.",
    "Researchers published findings on a breakthrough in quantum computing efficiency.",
    "Municipal elections were held peacefully with record turnout in three major cities.",
    "The agriculture department reported a 15 percent increase in wheat production this season.",
    "Hospital systems reported a decrease in emergency room visits following the public health campaign.",
    "The education ministry announced new curriculum standards for secondary schools nationwide.",
    "Diplomatic talks between the two countries resulted in a historic trade agreement.",
    "A new study in the New England Journal of Medicine highlights the benefits of Mediterranean diet.",
    "Stock markets closed higher on Friday after strong jobs data boosted investor confidence.",
    "The United Nations Security Council voted to extend peacekeeping operations in the region.",
    "Local authorities confirmed the arrest of three suspects in connection with the financial fraud case.",
    "The pharmaceutical company received regulatory approval for its new diabetes medication.",
    "Environmental regulators issued fines to the industrial plant for exceeding emission standards.",
    "The city council approved a budget of $3.2 billion for public infrastructure improvements.",
    "Scientists detected a fast radio burst from a source within our own Milky Way galaxy.",
    "The labor department published new workplace safety regulations after extensive public comment.",
    "Trade deficit narrowed in February as exports of goods and services increased significantly.",
    "Following the summit, both leaders agreed to a framework for reducing nuclear arsenals.",
    "Economists forecast moderate growth of 2.3 percent for the coming fiscal year.",
    "The transportation authority approved plans for a new light rail extension in the metropolitan area.",
    "Health officials confirmed the outbreak of influenza has peaked and cases are declining.",
    "The court dismissed the lawsuit citing lack of jurisdiction and insufficient evidence presented.",
    "Renewable energy sources accounted for over 20 percent of electricity generation last year.",
    "The national census bureau released preliminary population figures showing regional demographic shifts.",
    "Firefighters contained the wildfire after it burned approximately 12,000 acres in the national forest.",
    "The government published its annual report on human trafficking, noting a rise in prosecutions.",
    "Engineers completed the tunnel project three months ahead of the original construction schedule.",
    "The central statistics office reported that GDP grew by 1.8 percent in the third quarter.",
    "Authorities issued a public health advisory regarding elevated levels of pollen in the region.",
    "The defense ministry confirmed joint military exercises with allied nations scheduled for next month.",
    "Scientists used satellite data to map previously unknown ancient city ruins in the Amazon basin.",
    "A formal state visit between the two heads of government was announced for next spring.",
    "The national audit office released its annual review of government spending and efficiencies.",
    "Water treatment facilities in the affected area were restored following damage from the storm.",
    "The aviation authority approved new safety protocols for commercial drones operating in urban areas.",
    "A bipartisan committee released recommendations for reforming campaign finance laws.",
    "The ministry of education unveiled a new digital literacy program for students aged ten to fourteen.",
    "Border agency officials reported improved processing times after implementing new technology systems.",
]

# Fake news characteristics: sensational, unverified, conspiracy, emotional manipulation
FAKE_SAMPLES = [
    "SHOCKING: Government secretly putting microchips in vaccines to track citizens worldwide!",
    "BREAKING: Scientists confirm that the moon landing was staged in Hollywood studio in 1969!",
    "EXCLUSIVE: Deep state operatives behind major election fraud in all swing states revealed!",
    "BOMBSHELL: Celebrities are secretly lizard people controlling global financial systems!",
    "WARNING: 5G towers are spreading the coronavirus and causing mass sterilization in humans!",
    "URGENT: The cure for cancer discovered but Big Pharma is hiding it to maximize profits forever!",
    "EXPOSED: Obama was actually born in Kenya and his birth certificate was faked by CIA agents!",
    "ALERT: Chemtrails containing mind-control chemicals being sprayed over American cities daily!",
    "UNBELIEVABLE: Mainstream media is completely controlled by a secret globalist organization!",
    "MUST SHARE: Doctors never tell you this simple trick that reverses diabetes in just three days!",
    "OUTRAGE: George Soros funding antifa to overthrow the American government this summer!",
    "TRUTH REVEALED: The flat earth conspiracy finally proven by leaked NASA documents!",
    "DEVASTATING: Liberal elites running massive child trafficking rings from pizza restaurant basement!",
    "SECRET: The Illuminati controls every world leader and has been since ancient Babylon times!",
    "WAKE UP: Fluoride in drinking water is a government plot to make population docile and compliant!",
    "BANNED VIDEO: What they don't want you to see about the global depopulation agenda exposed!",
    "PROOF: Aliens helped build the pyramids and the Egyptian government is hiding the evidence!",
    "SHOCKING TRUTH: Hillary Clinton personally ordered the murder of four Benghazi whistleblowers!",
    "EXPOSED: COVID was bioweapon engineered in Chinese lab funded by Anthony Fauci secretly!",
    "BREAKING: Trump won the election in a landslide but computers flipped millions of votes!",
    "BOMBSHELL: Mainstream media suppressing evidence of massive voter fraud in all fifty states!",
    "URGENT WARNING: New world order agenda to implement global government by end of this decade!",
    "SCANDAL: Joe Biden has been secretly working for the Chinese Communist Party for thirty years!",
    "REVEALED: Doctors are paid to push dangerous vaccines and suppress natural immunity cures!",
    "SECRET AGENDA: Climate change hoax invented by globalists to impose carbon taxes on everyone!",
    "MUST READ: Deep state plotted assassination attempt on President that mainstream media covered!",
    "TRUTH BOMB: The Great Reset agenda will eliminate private property rights for all citizens!",
    "BREAKING: Global elite secretly meeting at Davos to plan economic collapse and reset currency!",
    "OUTRAGEOUS: George Soros paying protesters fifty dollars per hour to riot in American cities!",
    "EXPOSED: Bill Gates using vaccine program to depopulate Africa and reduce global population!",
    "ALERT: Social media censoring truth about election fraud to protect the deep state operatives!",
    "UNBELIEVABLE: Democrats plan to confiscate all guns within first hundred days of administration!",
    "BOMBSHELL EXCLUSIVE: Pentagon documents reveal secret alien contact program hidden from public!",
    "SHOCKING: Pizzagate documents proving elite pedophile ring leaked to internet and disappeared!",
    "URGENT: The Federal Reserve is privately owned by Rothschild family and controls all nations!",
    "BREAKING NEWS: Hunter Biden laptop contains evidence of treasonous crimes against entire nation!",
    "PROOF REVEALED: NASA has been hiding the existence of Planet X for more than thirty years!",
    "SECRET TRUTH: Masks cause oxygen deprivation and carbon dioxide poisoning with long term use!",
    "OUTRAGE: Mexican drug cartels controlling more than forty American cities with Democrat approval!",
    "EXPOSED: Antifa secretly trained in Cuba and funded by George Soros billions to destroy America!",
    "MUST SHARE BEFORE DELETED: Whistleblower reveals CIA mind control program is still operating!",
    "SHOCKING REVELATION: Princess Diana was murdered because she knew about royal family crimes!",
    "BOMBSHELL: Epstein list includes every major world leader and they murdered him to stay silent!",
    "WARNING: New legislation will allow government to forcibly vaccinate every American without consent!",
    "BREAKING: QAnon insider reveals deep state will be arrested in massive takedown very soon!",
    "EXPOSED: Hollywood celebrities performing satanic rituals at private Bohemian Grove gatherings!",
    "TRUTH: The entire COVID pandemic was planned and simulated by Bill Gates and WHO years ago!",
    "URGENT ALERT: Soros-funded judges releasing violent criminals to terrorize conservative cities!",
    "SCANDAL EXPOSED: Election machines in all battleground states were connected to the internet!",
    "MUST WATCH: Former CIA agent confesses chemtrails program is real and massive in scope!",
]

# ─── Build & Save Demo Model ──────────────────────────────────────────────────
def create_demo_model():
    print("🔧  Building demo model from synthetic data...")

    # Combine and label
    texts  = REAL_SAMPLES + FAKE_SAMPLES
    labels = [0] * len(REAL_SAMPLES) + [1] * len(FAKE_SAMPLES)

    # TF-IDF vectorizer
    def clean(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    cleaned = [clean(t) for t in texts]

    vectorizer = TfidfVectorizer(
        max_features = 5000,
        ngram_range  = (1, 2),
        stop_words   = "english",
        sublinear_tf = True,
    )
    X = vectorizer.fit_transform(cleaned)
    y = np.array(labels)

    # Logistic Regression
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X, y)

    # Quick self-check
    preds = model.predict(X)
    acc   = (preds == y).mean()
    print(f"   Training accuracy (demo data): {acc*100:.1f}%")

    # Save
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "model.pkl")
    vec_path   = os.path.join(base, "vectorizer.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"💾  Saved → {model_path}")
    print(f"💾  Saved → {vec_path}")
    print("\n✅  Demo model ready. Start Flask with: python app.py")
    print("⚠️   For production accuracy, run train_model.py with the ISOT dataset.")


if __name__ == "__main__":
    create_demo_model()
