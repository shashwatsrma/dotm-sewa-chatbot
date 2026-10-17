# Sample phrasings added to each chunk's embedded text.
# Add an entry here for every new chunk you create.
ALIASES = {
    "dotm_dl_renewal": "renew driving license expired license nabikaran | license renew garna | लाइसेन्स नवीकरण म्याद सकिएको",
    "dotm_dl_add_category": "add vehicle category to old license | category thap purano license ma | पुरानो लाइसेन्समा थप श्रेणी",
    "dotm_dl_categories": "license categories vehicle types bike scooter car jeep bus truck | kun gadi ko license kun category | सवारी प्रकार श्रेणी",
    "dotm_dl_warnings": "important warnings precautions mistakes before applying online | online form bharda dhyan dinu parne kura | चेतावनी सावधानी",
    "dotm_dl_new_office": "first time new driving license medical check at DoTM office | naya license office ma medical | नयाँ लाइसेन्स कार्यालयमा स्वास्थ्य परीक्षण",
    "dotm_dl_new_clinic": "first time new driving license medical check at clinic doctor | naya license clinic ma medical | नयाँ लाइसेन्स क्लिनिक चिकित्सक",
}

# Fake "off-topic" entries. If the closest match is one of these, reject the question.
ANCHORS = [
    "football cricket sports match score",
    "weather forecast rain temperature",
    "politics prime minister president government election",
    "cooking recipe food momo",
    "citizenship certificate passport tax and other government services",
    "movies music entertainment celebrity",
    "mobile phones laptops technology shopping",
    "bitcoin stock market money investment",
    "aja ko mausam pani parchha tapkram",
    "rastrapati pradhanmantri rajniti sarkar chunav",
    "khel football cricket match",
    "khana pakaune recipe momo",
    "खेल फुटबल क्रिकेट म्याच",
    "मौसम पानी तापक्रम",
    "राजनीति प्रधानमन्त्री राष्ट्रपति चुनाव",
    "खाना पकाउने रेसिपी",
    "general knowledge geography science history",
    "programming code computer help",
]