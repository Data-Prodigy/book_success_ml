import requests
import pandas as pd
import time

def fetch_novel_data():
    novel_df = pd.read_csv('Novel_Pred.csv')
    return novel_df

def keep_only_correct_variants(novel_df):
    """
    So I've got some records in my Title column that are a bit misspelled and yet match too closely so I can't use Fuzzy Wuzzy Process.extract, lol. 
    And I want to Keep only rows for specific titles IF their description matches exactly.
    All unrelated rows are preserved. 
    
    Parameters:
    - df: the original DataFrame with 'title' and 'description' columns
    - target_entries: a list of dicts with 'title' and 'description' keys

    Returns:
    - A cleaned DataFrame
    """
    target_entries = [{"title":"The Da Vinci Code","description":"""Harvard professor Robert Langdon receives an urgent late-night phone call while on business in Paris: the elderly curator of the Louvre, Jacques Sauni're, has been brutally murdered inside the museum. Alongside the body, police have found a series of baffling codes. As Langdon and a gifted French cryptologist, Sophie Neveu, begin to sort through the bizarre riddles, they are stunned to find a trail that leads to the works of Leonardo Da Vinci - and suggests the answer to a mystery that stretches deep into the vault of history. Langdon suspects the late curator was involved in the Priory of Sion - a centuries old secret society - and has sacrificed his life to protect the Priory's most sacred trust: the location of a vastly important religious relic hidden for centuries. But it now appears that Opus Dei, a clandestine sect that has long plotted to seize the Prirory's secret, has now made its move. Unless Langdon and Neveu can decipher the labyrinthine code and quickly assemble the pieces of the puzzle, the Priory's secret - and a stunning historical truth - will be lost forever. Breaking the mould of traditional suspense novels, The DA VINCI CODE is simultaneously lightning-paced, intelligent and intricately layered with remarkable research and detail. And in this exclusive edition Dan Brown allows the reader behind the scenes of the novel which now incorporates over 150 photographs and illustrations throughout the text showing the rich historical tapestry from which he drew his inspiration. The visual sources which provide both the backdrop and the stimulus for the novel's action are revealed for the first time and uniquely complement the reading experience."""},
                  {"title":"Angels & Demons","description":"""Experience the explosive, â€œintriguing, imaginative, and very suspensefulâ€ (Dale Brown, New York Times bestselling author) classic thriller from the #1 New York Times bestselling author of The Da Vinci Code and Inferno that follows Robert Langdon on a white-knuckled race against time to uncover the darkest secrets of Rome. An ancient secret brotherhood. A devastating new weapon of destruction. An unthinkable target. When world-renowned Harvard symbologist Robert Langdon is summoned to a Swiss research facility to analyze a mysterious symbol, he discovers evidence of the unimaginable: the resurgence of an ancient and powerful secret brotherhood known as the Illuminatiâ€”which has its sights on its longtime enemy: the Catholic Church. Desperate to save the Vatican, Langdon joins forces in Rome with the beautiful and mysterious scientist Vittoria Vetra. Together, they embark on a frantic hunt through sealed crypts, dangerous catacombs, deserted cathedrals, and the most secretive vault on earthâ€”the long-forgotten Illuminati lair which houses the only hope for the salvation of the Catholic Church. â€œA breathless, real-time adventureâ€ (San Francisco Chronicle), Angels & Demons is an unputdownable and whip-smart thriller that careens from enlightening epiphanies to dark truths as the battle between science and religion turns to war."""},
                  {"title":"Inferno","description":"""NOW A MAJOR MOTION PICTURE With the publication of his groundbreaking novels The Da Vinci Code, The Lost Symbol, and Angels & Demons, Dan Brown has become an international bestselling sensation, seamlessly fusing codes, symbols, art, and history into riveting thrillers that have captivated hundreds of millions of readers around the world. Now, Dan Brown takes readers deep into the heart of Italy . . . guiding them through a landscape that inspired one of historyâ€™s most ominous literary classics. â€œThe darkest place in hell are reserved for those who maintain their neutrality in times of moral crisis.â€ Harvard professor of symbology Robert Langdon awakens in a hospital in the middle of the night. Disoriented and suffering from a head wound, he recalls nothing of the last thirty-six hours, including how he got there . . . or the origin of the macabre object that his doctors discover hidden in his belongings. Langdonâ€™s world soon erupts into chaos, and he finds himself on the run in Florence with a stoic young woman, Sienna Brooks, whose clever maneuvering saves his life. Langdon quickly realizes that he is in possession of a series of disturbing codes created by a brilliant scientistâ€”a genius whose obsession with the end of the world is matched only by his passion for one of the most influential masterpieces ever writtenâ€”Dante Alighieriâ€™s dark epic poem The Inferno. Racing through such timeless locations as the Palazzo Vecchio, the Boboli Gardens, and the Duomo, Langdon and Brooks discover a network of hidden passageways and ancient secrets, as well as a terrifying new scientific paradigm that will be used either to vastly improve the quality of life on earth . . . or to devastate it. In his most riveting and thought-provoking novel to date, Dan Brown has raised the bar yet again. Inferno is a sumptuously entertaining readâ€”a novel that will captivate readers with the beauty of classical Italian art, history, and literature . . . while also posing provocative questions about the role of cutting-edge science in our future."""},
                  {"title":"Deception Point","description":"""From the #1 New York Times bestselling author of The Da Vinci Code, Angels & Demons, and Inferno and the â€œmaster of smart thrillsâ€ (People) comes a â€œrocket-fast thrillerâ€ (Vince Flynn) about an astonishing NASA discovery that unravels a deadly conspiracy that leads all the way to the White House. When a NASA satellite spots evidence of an astoundingly rare object buried deep in the Arctic ice, the floundering space agency proclaims a much-needed victoryâ€”one that could have profound implications for US space policy and the impending presidential election. With his re-election hanging in the balance, the President sends White House Intelligence analyst Rachel Sexton to the Milne Ice Shelf to verify the authenticity of the find. Accompanied by a team of experts, including the charismatic academic Michael Tolland, Rachel uncovers the unthinkable: evidence of scientific trickery. Before she can contact the President, she and Michael are attacked by a deadly team of assassins controlled by a mysterious power broker who will stop at nothing to hide the truth. Fleeing for their lives in an environment as desolate as it is lethal, their only hope for survival is to find out who is behind this masterful ploy. The truth, they will learn, is the most shocking deception of all in this â€œtaut, fast-paced, barn-burner of a bookâ€ (St. Petersburg Times)."""},
                  {"title":"The Lost Symbol","description":"""Robert Langdon returns in Dan Brown's brilliant new thriller, THE SECRET OF SECRETS, available for pre-order now. The bestselling continuation of global phenomenon THE DA VINCI CODE, featuring symbologist Robert Langdon. The Capitol Building, Washington DC: Harvard symbologist Robert Langdon believes he is here to give a lecture. He is wrong. Within minutes of his arrival, a shocking object is discovered. It is a gruesome invitation into an ancient world of hidden wisdom. When Langdon's mentor, Peter Solomon - prominent mason and philanthropist - is kidnapped, Langdon realizes that his only hope of saving his friend's life is to accept this mysterious summons. It is to take him on a breathless chase through Washington's dark history. All that was familiar is changed into a shadowy, mythical world in which Masonic secrets and never-before-seen revelations seem to be leading him to a single impossible and inconceivable truth..."""},
                  {"title":"Origin","description":"""Robert Langdon returns in Dan Brown's brilliant new thriller, THE SECRET OF SECRETS, available for pre-order now. The global bestseller featuring symbologist Robert Langdon, from the author of THE DA VINCI CODE and 'master of the intellectual cliffhanger' (Wall Street Journal) Dan Brown. Robert Langdon, Harvard professor of symbology and religious iconology, arrives at the Guggenheim Museum Bilbao to attend the unveiling of an astonishing scientific breakthrough. The eveningâ€™s host is billionaire Edmond Kirsch, a futurist whose dazzling high-tech inventions and audacious predictions have made him a controversial figure around the world. But Langdon and several hundred guests are left reeling when the meticulously orchestrated evening is suddenly blown apart. There is a real danger that Kirschâ€™s precious discovery may be lost in the ensuing chaos. With his life under threat, Langdon is forced into a desperate bid to escape Bilbao, taking with him the museumâ€™s director, Ambra Vidal. Together they flee to Barcelona on a perilous quest to locate a cryptic password that will unlock Kirschâ€™s secret. To evade a devious enemy who is one step ahead of them at every turn, Langdon and Vidal must navigate the labyrinthine passageways of extreme religion and hidden history. On a trail marked only by enigmatic symbols and elusive modern art, Langdon and Vidal will come face-to-face with a breathtaking truth that has remained buried â€“ until now."""},
                  {"title":"Digital Fortress","description":"""A former National Security Agency programmer threatens to release a mathematical formula that will allow organized crime and terrorism to skyrocket, unless the code-breaking computer that is used to keep them in check--but that violates civil rights--is not exposed to the public."""},
                  {"title":"Things Fall Apart","description":"""â€œA true classic of world literature . . . A masterpiece that has inspired generations of writers in Nigeria, across Africa, and around the world.â€ â€”Barack Obama â€œAfrican literature is incomplete and unthinkable without the works of Chinua Achebe.â€ â€”Toni Morrison Nominated as one of Americaâ€™s best-loved novels by PBSâ€™s The Great American Read Things Fall Apart is the first of three novels in Chinua Achebe's critically acclaimed African Trilogy. It is a classic narrative about Africa's cataclysmic encounter with Europe as it establishes a colonial presence on the continent. Told through the fictional experiences of Okonkwo, a wealthy and fearless Igbo warrior of Umuofia in the late 1800s, Things Fall Apart explores one man's futile resistance to the devaluing of his Igbo traditions by British political andreligious forces and his despair as his community capitulates to the powerful new order. With more than 20 million copies sold and translated into fifty-seven languages, Things Fall Apart provides one of the most illuminating and permanent monuments to African experience. Achebe does not only capture life in a pre-colonial African village, he conveys the tragedy of the loss of that world while broadening our understanding of our contemporary realities."""},
                  {"title":"Arrow of God","description":"""Set in the Ibo heartland of eastern Nigeria, one of Africa's best-known writers describes the conflict between old and new in its most poignant aspect: the personal struggle between father and son. The third book in Achebe's "African Trilogy", following Things Fall Apart and No Longer at Ease, Arrow of God is the story of Ezeulu, the chief priest of several villages who wrestles with colonial powers as he butts heads with Christian missionaries dispatched to the area. A fictional discussion of Colonial rule in 1920's Nigeria, Achebe brings religion and family relations into a discussion of politics and national identity."""},
                  {"title":"There Was a Country","description":"""From the legendary author of Things Fall Apart comes this long-awaited memoir recalling Chinua Achebe's personal experiences of and reflections on the Biafran War, one of Nigeria's most tragic civil wars Chinua Achebe, the author of Things Fall Apart, was a writer whose moral courage and storytelling gifts have left an enduring stamp on world literature. There Was a Country was his long-awaited account of coming of age during the defining experience of his life: the Nigerian Civil War, also known as the Biafran War of 1967-1970. It became infamous around the world for its impact on the Biafrans, who were starved to death by the Nigerian government in one of the twentieth century's greatest humanitarian disasters. Caught up in the atrocities were Chinua Achebe and his young family. Achebe, already a world-renowned novelist, served his Biafran homeland as a roving cultural ambassador, witnessing the war's full horror first-hand. Immediately after the war, he took an academic post in the United States, and for over forty years he maintained a considered silence on those terrible years, addressing them only obliquely through his poetry. After years in the making There Was A Country presents his towering reckoning with one of modern Africa's most fateful experiences, both as he lived it and came to understand it. Marrying history and memoir, with the author's poetry woven throughout, There Was a Country is a distillation of vivid observation and considered research and reflection. It relates Nigeria's birth pangs in the context of Achebe's own development as a man and a writer, and examines the role of the artist in times of war. Reviews: 'No writer is better placed than Chinua Achebe to tell the story of the Nigerian Biafran war ... [The book] makes you pine for the likes of Achebe to govern ... We have in There Was a Country an elegy from a master storyteller who has witnessed the undulating fortunes of a nation' Noo Saro-Wiwa, Guardian 'Chinua Achebe's history of Biafra is a meditation on the condition of freedom. It has the tense narrative grip of the best fiction. It is also a revelatory entry into the intimate character of the writer's brilliant mind and bold spirit. Achebe has created here a new genre of literature' Nadine Gordimer 'Part-history, part-memoir, [Achebe's] moving account of the war is laced with anger, but there is also an abiding tone of regret for what Nigeria might have been without conflict and mismanagement' Sunday Times About the author: Chinua Achebe was born in Nigeria in 1930. He published novels, short stories, essays, and children's books. His volume of poetry, Christmas in Biafra, was the joint winner of the first Commonwealth Poetry Prize. Of his novels, Arrow of God won the New Statesman-Jock Campbell Award, and Anthills of the Savannah was a finalist for the 1987 Booker Prize. Things Fall Apart, Achebe's masterpiece, has been published in fifty different languages and has sold more than ten million copies. Achebe lectured widely, receiving many honors from around the world. He was the recipient of the Nigerian National Merit Award, Nigeria's highest award for intellectual achievement. In 2007, he won the Man Booker International Prize. He died in 2013."""},
                  {"title":"A Man of the People","description":"""By the renowned author of Things Fall Apart, this novel foreshadows the Nigerian coups of 1966 and shows the color and vivacity as well as the violence and corruption of a society making its own way between the two worlds. In the landscape of Western Africa, two political traditions collide: the old bush politians against the new intelellectual generation, and a mentor and his protegee must wage the war. Achebe details one society's struggle with the inner turmoil created in the wake of the new-found freedom from the colonial order. This is a story about national identity and political unity."""},
                  {"title":"Anthills of the Savannah","description":"""A searing satire of political corruption and social injustice from the celebrated author of Things Fall Apart "Achebe has written a story that sidesteps both ideologies of the African experience and political agendas, in order to lead us to a deeply human universal wisdom." â€”Washington Post Book World In the fictional West African nation of Kangan, newly independent of British rule, the hopes and dreams of democracy have been quashed by a fierce military dictatorship. Chris Oriko is a member of the president's cabinet for life, and one of the leader's oldest friends. When the president is charged with censoring the opportunistic editor of the state-run newspaper--another childhood friend--Chris's loyalty and ideology are put to the test. The fate of Kangan hangs in the balance as tensions rise and a devious plot is set in motion to silence a firebrand critic. From Chinua Achebe, the legendary author of Things Fall Apart, Anthills of the Savannah is "A vision of social change that strikes us with the force of prophecy." (USA Today)"""},
                  {"title":"Chike and the River","description":"""After an 11-year-old Nigerian boy leaves his small village to live with his uncle in the city, he is exposed to a range of new experiences and becomes fascinated with crossing the Niger River on a ferry boat."""},
                  {"title":"Hopes and Impediments","description":"""The author's critical writings over the past twenty-five years use his creative energies to expose the monster of racist habit and offer a new perspective on the human condition."""},
                  {"title":"The Flute","description":"""EventyrfortÃ¦lling fra Nigeria om en dreng, hvis flÃ¸jte bringer rigdom"""},
                  {"title":"How the Leopard Got His Claws","description":"""Recounts how the leopard got his claws and teeth and why he rules the forest with terror."""},
                  {"title":"Purple Hibiscus","description":"""Fifteen-year-old Kambili's world is circumscribed by the high walls of her family compound and the frangipani trees she can see from her bedroom window. Her wealthy Catholic father, although generous and well-respected, is repressive and fanatically religious. Her life is lived under his shadow and regulated by schedules: prayer, sleep, study, and more prayer. When Nigeria begins to fall apart under a military coup, Kambili's father, involved in mysterious ways with the unfolding political crisis, sends Kambili and her brother away to their aunt's. The house is noisy and full of laughter. Here she discovers love and a life beyond the confines of her father's authority."""},
                  {"title":"Dream Count","description":"""A publishing event ten years in the makingâ€”a searing, exquisite new novel by the bestselling and award-winning author of Americanah and We Should All Be Feministsâ€”the story of four women and their loves, longings, and desires Chiamaka is a Nigerian travel writer living in America. Alone in the midst of the pandemic, she recalls her past lovers and grapples with her choices and regrets. Zikora, her best friend, is a lawyer who has been successful at everything untilâ€”betrayed and brokenheartedâ€”she must turn to the person she thought she needed least. Omelogor, Chiamakaâ€™s bold, outspoken cousin, is a financial powerhouse in Nigeria who begins to question how well she knows herself. And Kadiatou, Chiamakaâ€™s housekeeper, is proudly raising her daughter in Americaâ€”but faces an unthinkable hardship that threatens all she has worked to achieve. In Dream Count, Adichie trains her fierce eye on these women in a sparkling, transcendent novel that takes up the very nature of love itself. Is true happiness ever attainable or is it just a fleeting state? And how honest must we be with ourselves in order to love, and to be loved? A trenchant reflection on the choices we make and those made for us, on daughters and mothers, on our interconnected world, Dream Count pulses with emotional urgency and poignant, unflinching observations of the human heart, in language that soars with beauty and power. It confirms Adichieâ€™s status as one of the most exciting and dynamic writers on the literary landscape."""},
                  {"title":"Americanah","description":"""10th ANNIVERSARY EDITION â€¢ NATIONAL BESTSELLER â€¢ A modern classic about star-crossed lovers that explores questions of race and being Black in Americaâ€”and the search for what it means to call a place home. â€¢ From the award-winning author of We Should All Be Feminists and Half of a Yellow Sun â€¢ WITH A NEW INTRODUCTION BY THE AUTHOR "An expansive, epic love story."â€”O, The Oprah Magazine One of the New York Timesâ€™s 100 Best Books of the 21st Century â€¢ One of The Atlanticâ€™s Great American Novels of the Past 100 Years Ifemelu and Obinze are young and in love when they depart military-ruled Nigeria for the West. Beautiful, self-assured Ifemelu heads for America, where despite her academic success, she is forced to grapple with what it means to be Black for the first time. Quiet, thoughtful Obinze had hoped to join her, but with postâ€“9/11 America closed to him, he instead plunges into a dangerous, undocumented life in London. At once powerful and tender, Americanah is a remarkable novel that is "dazzlingâ€¦funny and defiant, and simultaneously so wise." â€”San Francisco Chronicle"""},
                  {"title":"The Thing Around Your Neck","description":"""From Chimamanda Ngozi Adichie, the Orange Prize-winning author of Half of a Yellow Sun, come twelve dazzling stories in which she turns her penetrating eye on the ties that bind men and women, parents and children, Nigeria and the West. In 'A Private Experience,' a medical student hides from a violent riot with a poor Muslim woman whose dignity and faith force her to confront the realities and fears she's been pushing away. In 'Tomorrow Is Too Far,' a woman unlocks the devastating secret that surrounds her brother's death. The young mother at the center of 'Imitation' finds her comfortable life threatened when she learns that her husband back in Lagos has moved his mistress into their home. And the title story depicts the choking loneliness of a Nigerian girl who moves to an America that turns out to be nothing like the country she expected; though falling in love brings her desires nearly within reach, a death in her homeland forces her to re-examine them. Searing and profound, suffused with beauty, sorrow and longing, this collection is a resounding confirmation of Chimamanda Ngozi Adichie's prodigious storytelling powers."""},
                  {"title":"Imitation","description":"""Nkem is living a life of wealth and security in America, until she discovers that her husband is keeping a girlfriend back home in Nigeria. In this high-intensity story of passion and the masks we all wear, Chimamanda Ngozi Adichie, author of the acclaimed novels Half of a Yellow Sun and Americanah and winner of the Orange Prize and the National Book Critics Circle Award, explores the ties that bind men and women, parents and children, Africa and the United States. â€œImitationâ€ is a selection from Adichieâ€™s collection The Thing Around Your Neck. An eBook short."""},
                  {"title":"We Should All Be Feminists: the Desk Diary 2021","description":"""A beautiful hardback, elastic hinged desk diary with a week to a view alongside an inspiring and powerful quote or a photograph of Chimamanda and a brand-new introduction from her. 'We teach girls to shrink themselves, to make themselves smaller.' 'Not one day longer.' This year, with some words of wisdom to inspire you, you will walk tall. Make 2021 your biggest year yet, with this beautifully designed hardback diary filled with some of Chimamanda Ngozi Adichie's most inspirational quotes. From her award-winning novels like Half of a Yellow Sun and Americanah, to her stirring calls to arms We Should All Be Feminists and Dear Ijeawele, from her countless magazine covers, her work with BeyoncÃ© and sharing the stage with Michelle Obama, Chimamanda Ngozi Adichie is one of the most defining and stirring voices of our time - a truly modern icon. Now, each day, Adichie will inspire you to stand up and be heard. Start your year off on the right foot and be inspired to be exactly who you want to be in 2021. After all, as Chimamanda says: 'It's not your job to be likeable. It's your job to be yourself.'"""},
                  {"title":"Dear Ijeawele, Or a Feminist Manifesto in Fifteen Suggestions","description":"""From the best-selling author of Americanah and We Should All Be Feminists comes a powerful new statement about feminism today - written as a letter to a friend."""},
                  {"title":"Notes on Grief","description":"""A devastating essay on loss and the people we love from Chimamanda Ngozi Adichie, the bestselling author of Americanah and Half of a Yellow Sun."""},
                  {"title":"Nineteen Eighty-four","description":"""(Book Jacket Status: Jacketed) "Nineteen Eighty-Four" revealed George Orwell as one of the twentieth century's greatest mythmakers. While the totalitarian system that provoked him into writing it has since passed into oblivion, his harrowing cautionary tale of a man trapped in a political nightmare has had the opposite fate: its relevance and power to disturb our complacency seem to grow decade by decade. In Winston Smith's desperate struggle to free himself from an all-encompassing, malevolent state, Orwell zeroed in on tendencies apparent in every modern society, and made vivid the universal predicament of the individual."""},
                  {"title":"Animal Farm a Fairy Story","description":"""Animal Farm is an allegorical novella by George Orwell, first published in England on 17 August 1945. The book tells the story of a group of farm animals who rebel against their human farmer, hoping to create a society where the animals can be equal, free, and happy. Ultimately, however, the rebellion is betrayed, and the farm ends up in a state as bad as it was before, under the dictatorship of a pig named Napoleon. According to Orwell, the fable reflects events leading up to the Russian Revolution of 1917 and then on into the Stalinist era of the Soviet Union. Orwell, a democratic socialist, was a critic of Joseph Stalin and hostile to Moscow-directed Stalinism, an attitude that was critically shaped by his experiences during the Spanish Civil War. The Soviet Union, he believed, had become a brutal dictatorship built upon a cult of personality and enforced by a reign of terror. In a letter to Yvonne Davet, Orwell described Animal Farm as a satirical tale against Stalin ("un conte satirique contre Staline"), and in his essay "Why I Write" (1946), wrote that Animal Farm was the first book in which he tried, with full consciousness of what he was doing, "to fuse political purpose and artistic purpose into one whole"."""},
                  {"title":"Homage to Catalonia","description":"""In "Homage to Catalonia," George Orwell delivers a poignant firsthand account of his experiences fighting in the Spanish Civil War, encompassing not only the brutality and chaos of war but also the complex political dynamics at play. Written in a straightforward yet evocative prose style, the narrative traverses Orwell's initial enthusiasm for the anti-fascist cause, his disillusionment with the infighting among leftist factions, and his critical reflections on the nature of totalitarianism. The book serves as both a travel memoir and a political treatise, immersing readers in the visceral reality of war while critically examining the ideological battles that shaped Europe during the 20th century. George Orwell, a prominent literary figure known for his sharp political insight and commitment to social justice, wrote this work following his return from Spain in the late 1930s. His engagement in the war profoundly influenced his worldview, fortifying his anti-totalitarian beliefs and illuminating the struggles against oppression, themes that would later resonate in his more famous works such as "1984" and "Animal Farm." Orwell's personal experiences allow him to navigate the emotional and political landscape of the conflict with authenticity and depth. "Homage to Catalonia" is an essential read for those interested in the interplay between democracy and totalitarianism, as well as for anyone seeking a deeper understanding of the Spanish Civil War's impact on modern politics and literature. Orwell's incisive observations offer enduring lessons about the fragility of freedom and the complexities of ideological commitment, making this book a vital contribution to both historical and literary scholarship."""},
                  {"title":"Down and Out in Paris and London","description":"""In "Down and Out in Paris and London," George Orwell intricately weaves a narrative that explores the harrowing realities of poverty and social injustice through his keen observations of life on the margins. The book oscillates between the bustling streets of Paris and the grim atmospheres of London, capturing the essence of transient existence in a tone that is both poignant and satirical. Through vivid descriptions and a memoir-like style, Orwell immerses the reader in the struggles of the destitute, providing a profound social commentary that reflects the socio-economic disparities of the early 20th century. George Orwell, born Eric Arthur Blair, was profoundly influenced by his experiences in the working class, which led him to adopt a politically charged perspective throughout his literary career. His time spent in the slums of London and the impoverished quarters of Paris instilled a deep empathy for the downtrodden, motivating him to highlight the human condition. This autobiographical account not only reveals Orwell's disdain for capitalism but also showcases his commitment to truth and transparency in writing. "Down and Out in Paris and London" is essential reading for those seeking to understand the societal structures that perpetuate inequality. With its unflinching honesty and powerful insight, it invites readers to confront the uncomfortable realities of poverty, making it a compelling and thought-provoking exploration that resonates well beyond its time."""},
                  {"title":"Keep the Aspidistra Flying","description":""""Keep the Aspidistra Flying", first published in 1936, is a socially critical novel by George Orwell. It is set in 1930s London. The main theme is Gordon Comstock's romantic ambition to defy worship of the money-god and status, and the dismal life that results. (Wikipedia)"""},
                  {"title":"Burmese Days","description":""""Burmese Days" is the first novel by English writer George Orwell, published in 1934. Set in British Burma during the waning days of Empire, when Burma was ruled from Delhi as part of British India, it is "a portrait of the dark side of the British Raj". At the centre of the novel is John Flory, "the lone and lacking individual trapped within a bigger system that is undermining the better side of human nature". The novel describes "both indigenous corruption and imperial bigotry" in a society where, "after all, natives were nativesâ€”interesting, no doubt, but finally...an inferior people". "Burmese Days" was first published "further afield", in the United States, because of concerns that it might be potentially libelous; that the real provincial town of Katha had been described too realistically; and that some of its fictional characters were based too closely on identifiable people. A British edition, with altered names, appeared a year later. Nonetheless, Orwell's harsh portrayal of colonial society was felt by "some old Burma hands" to have "rather let the side down". In a letter from 1946, Orwell wrote, "I dare say it's unfair in some ways and inaccurate in some details, but much of it is simply reporting what I have seen"."""},
                  {"title":"Coming Up for Air","description":"""Before the war! How long shall we go on saying that, I wonder? How long before the answer will be 'Which war?' The approach of the Second World War finds suburban insurance agent George Bowling in a reflective mood. As he thinks back to the sedate Oxfordshire village of his Edwardian boyhood, he contemplates regretfully what has happened to England since then, from the First World War, in which he served, to the seemingly inescapable money-grubbing and mechanization of everyday life in modern London. A lucky windfall allows Bowling to make a secret return to his idyllic birthplace: a fortifying respite, he hopes, from the struggles of life in a modern city on the verge of war. But is there really any going back? Published in 1939, Coming Up for Air is the most accomplished of Orwell's early realist novels, casting light on the development of Orwell's distinctive thinking as a cultural critic. The novel explores many of the themes Orwell later reprised in 1984: nostalgia, memory, and disillusionment in the face of modernity's ills, including industrialisation, capitalist exploitation, and endless war."""},
                  {"title":"A Time to Kill","description":"""#1 NEW YORK TIMES BESTSELLER â€¢ The master of the legal thriller probes the savage depths of racial violence in this searing courtroom drama featuring the beloved Jake Brigance. â€œJohn Grisham may well be the best American storyteller writing today.â€â€”The Philadelphia Inquirer The life of a ten-year-old black girl is shattered by two drunken and remorseless white men. The mostly white town of Clanton in Ford County, Mississippi, reacts with shock and horror at the inhuman crimeâ€”until the girlâ€™s father acquires an assault rifle and takes justice into his own hands. For ten days, as burning crosses and the crack of sniper fire spread through the streets of Clanton, the nation sits spellbound as defense attorney Jake Brigance struggles to save his clientâ€™s lifeâ€”and then his own. Donâ€™t miss any of John Grishamâ€™s gripping books featuring Jake Brigance: A TIME TO KILL â€¢ SYCAMORE ROW â€¢ A TIME FOR MERCY â€¢ SPARRING PARTNERS"""},
                  {"title":"The Firm","description":"""He thought it was his dream job... It turned into his worst nightmare. When Mitch McDeere qualified third in his class at Harvard, offers poured in from every law firm in America. Bendini, Lambert and Locke were a small, well-respected firm, but their offer exceeded Mitch's wildest expectations- a fantastic salary, a new home, and the keys to a brand new BMW. Except for the mysterious deaths of previous lawyers with the firm. And the FBI investigations. And the secret files. Mitch soon realises that he's working for the Mafia's law firm, and there's no way out - because you don't want this company's severance package. To survive, he'll have to play both sides against each other... and navigate a vast criminal conspiracy that goes higher than he ever imagined."""},
                  {"title":"The Pelican Brief","description":"""In suburban Georgetown a killer's Reeboks whisper on the front floor of a posh home... In a seedy D.C. porno house a patron is swiftly garroted to death... The next day America learns that two of its Supreme Court justices have been assassinated. And in New Orleans, a young law student prepares a legal brief... To Darby Shaw it was no more than a legal shot in the dark, a brilliant guess. To the Washington establishment it was political dynamite. Suddenly Darby is witness to a murder -- a murder intended for her. Going underground, she finds there is only one person she can trust -- an ambitious reporter after a newsbreak hotter than Watergate -- to help her piece together the deadly puzzle. Somewhere between the bayous of Louisiana and the White House's inner sanctums, a violent cover-up is being engineered. For somone has read Darby's brief. Someone who will stop at nothing to destroy the evidence of an unthinkable crime. "From the Paperback edition."""},
                  {"title":"The Client","description":"""#1 NEW YORK TIMES BESTSELLER â€¢ A white-knuckle legal thriller that begins in a weedy lot on the outskirts of Memphis, when two boys watch a shiny Lincoln pull up to the curb. . . . â€œThe desire to find out what happens next keeps the reader turning the pages. Grisham is an absolute master of the chase story.â€â€”The Washington Post Eleven-year-old Mark Sway and his younger brother were sharing a forbidden cigarette when a chance encounter with a suicidal lawyer left Mark knowing a bloody and explosive secret: the whereabouts of the most sought-after dead body in America. Now Mark is caught between a legal system gone mad and a mob killer desperate to cover up his crime. And his only ally is a woman named Reggie Love, who has been a lawyer for all of four years. Prosecutors are willing to break all the rules to make Mark talk. The mob will stop at nothing to keep him quiet. And Reggie will do anything to protect her clientâ€”even take a last, desperate gamble that could win Mark his freedom . . . or cost them both their lives."""},
                  {"title":"The Rainmaker","description":"""Abandoning his aspirations about living the good life, Memphis attorney Rudy Baylor fears he will lose everything in the face of a pivotal case that could cost him his life or turn him into the biggest rainmaker in the land. Reprint."""},
                  {"title":"The Runaway Jury","description":"""#1 NEW YORK TIMES BESTSELLER They are at the center of a multimillion-dollar legal hurricane: twelve men and women who have been investigated, watched, manipulated, and harassed by high-priced lawyers and consultants who will stop at nothing to secure a verdict. Now the jury must make a decision in the most explosive civil trial of the century, a precedent-setting lawsuit against a giant tobacco company. But only a handful of people know the truth: that this jury has a leader, and the verdict belongs to him. He is known only as Juror #2. But he has a name, a past, and he has planned his every move with the help of a beautiful woman on the outside. Now, while a corporate empire hangs in the balance, while a grieving family waits, and while lawyers are plunged into a battle for their careers, the truth about Juror #2 is about to explode in a cross fire of greed and corruption--and with justice fighting for its life."""},
                  {"title":"The Testament","description":"""#1 "NEW YORK TIMES" BESTSELLER In a plush Virginia office, a rich, angry old man is furiously rewriting his will. With his death just hours away, Troy Phelan wants to send a message to his children, his ex-wives, and his minions--a message that will touch off a vicious legal battle and transform dozens of lives. Because Troy Phelan's new will names a sole surprise heir to his eleven-billion-dollar fortune: a mysterious woman named Rachel Lane, a missionary living deep in the jungles of Brazil. Enter the lawyers. Nate O'Riley is fresh out of rehab, a disgraced corporate attorney handpicked for his last job: to find Rachel Lane at any cost. As Phelan's family circles like vultures in D.C., Nate goes crashing through the Brazilian jungle, entering a world where money means nothing, where death is just one misstep away, and where a woman--pursued by enemies and friends alike--holds a stunning surprise of her own."""},
                  {"title":"The Street Lawyer","description":"""Michael Brock is billing the hours, making the money, rushing relentlessly to the top of Drake & Sweeney, a giant D.C. law firm. One step away from partnership, Michael has it all. Then, in an instant, it all comes apart."""},
                  {"title":"The Whistler","description":"""#1 NEW YORK TIMES BESTSELLER â€¢ A high-stakes thrill ride through the darkest corners of the Sunshine State, from the author hailed as â€œthe best thriller writer aliveâ€ by Ken Follett We expect our judges to be honest and wise. Their integrity is the bedrock of the entire judicial system. We trust them to ensure fair trials, to protect the rights of all litigants, to punish those who do wrong, and to oversee the flow of justice. But what happens when a judge bends the law or takes a bribe? Lacy Stoltz is an investigator for the Florida Board on Judicial Conduct. It is her job to respond to complaints dealing with judicial misconduct. After nine years with the Board, she knows that most problems are caused by incompetence, not corruption. But a corruption case eventually crosses her desk. A previously disbarred lawyer is back in business, and he claims to know of a Florida judge who has stolen more money than all other crooked judges combined. And not just crooked judges in Florida. All judges, from all states, and throughout United States history. And now he wants to put a stop to it. His only client is a person who knows the truth and wants to blow the whistle and collect millions under Florida law. When the case is assigned to Lacy, she immediately suspects that this one could be dangerous. Dangerous is one thing. Deadly is something else. â€œ[A] main character [whoâ€™s] a seriously appealing woman . . . a whistle-blower who secretly calls attention to corruption . . . a strong and frightening sense of place . . . [John Grishamâ€™s] on his game.â€â€”Janet Maslin, The New York Times â€œ[John Grisham is] our guide to the byways and backwaters of our legal system, superb in particular at ferreting out its vulnerabilities and dramatizing their abuse in gripping style.â€â€”USA Today â€œRiveting . . . an elaborate conspiracy.â€â€”The New York Times Book Review Donâ€™t miss John Grishamâ€™s new book, THE EXCHANGE: AFTER THE FIRM!"""},
                  {"title":"The Reckoning","description":"""#1 NEW YORK TIMES BEST SELLER â€¢ John Grisham's most powerful, surprising, and suspenseful thriller yet â€¢ â€œA murder mystery, a courtroom drama, a family saga.â€ â€”USA Today October 1946, Clanton, Mississippi Pete Banning was Clanton, Mississippiâ€™s favorite sonâ€”a decorated World War II hero, the patriarch of a prominent family, a farmer, father, neighbor, and a faithful member of the Methodist church. Then one cool October morning he rose early, drove into town, and committed a shocking crime. Pete's only statement about itâ€”to the sheriff, to his lawyers, to the judge, to the jury, and to his familyâ€”was: "I have nothing to say." He was not afraid of death and was willing to take his motive to the grave. In a major novel unlike anything he has written before, John Grisham takes us on an incredible journey, from the Jim Crow South to the jungles of the Philippines during World War II; from an insane asylum filled with secrets to the Clanton courtroom where Peteâ€™s defense attorney tries desperately to save him. Reminiscent of the finest tradition of Southern Gothic storytelling, The Reckoning would not be complete without Grishamâ€™s signature layers of legal suspense, and he delivers on every page."""},
                  {"title":"The Brethren","description":"""Trumble is a minimum-security federal prison, home to drug dealers, bank robbers and three former judges known as The Brethren. Each day in the law library, The Brethren spend hours fine-tuning their mail scam, and it's starting to pay big. But when their scam goes awry and ensnares the wrong victim - a powerful man on the outside - they're in trouble. Because this man has dangerous friends. Just as the prison failed to protect society from The Brethren, it won't protect them from their victims... _____________________________ 'A master at the art of deft characterisation and the skilful delivery of hair-raising crescendos' - Irish Independent 'John Grisham is the master of legal fiction' - Jodi Picoult 'The best thriller writer alive' - Ken Follett 'John Grisham has perfected the art of cooking up convincing, fast-paced thrillers' - Telegraph 'Grisham is a superb, instinctive storyteller' - The Times 'Grisham's storytelling genius reminds us that when it comes to legal drama, the master is in a league of his own.' - Daily Record 'Masterful - when Grisham gets in the courtroom he lets rip, drawing scenes so real they're not just alive, they're pulsating' - Mirror 'A giant of the thriller genre' - TimeOut John Grisham, Sunday Times bestseller, December 2023"""},
                  {"title":"The Litigators","description":"""Law firm partners Oscar Finley and Wally Figg see a chance for huge financial gain when they learn of a pending class action lawsuit against the makers of Krayoxx, a popular cholesterol-reducing drug suspected of causing heart attacks."""},
                  {"title":"Carrie","description":"""Make a date with terror -- and live the nightmare that is ..."""},
                  {"title":"The Shining","description":"""This inspiring and compelling book has won ten awards to date, including Honorable Mentions at the December 2012 New England and London Book Festivals, October 2012 Southern California Book Festival and June 2012 New York Book Festival in the category of Spiritual books; is winner of the North American Bookdealers Ì Exchange (NABE) Pinnacle Award for â€œInspirationalâ€ books in Spring 2011; and has become a much sought-after reference for people seeking to affect positive change around the globe. Readers are: * taught how to recognize, harness and channel positive personal power for the betterment of themselves, their loved ones, associates and our universe * provided an invaluable checklist of great leadership behaviors and attitudes * taught how to recognize controlling behaviors of others and the negative patterns in society * inspired to be the best they can be * compelled to ask themselves "why am I here; what good can I do for humanity?" * taught how to recognize a self-limiting posture so they can improve their level of self-awareness......to the point of real self-intelligence and, by so doing, break free of lifeâ€™s boxes, labels and restrictions * encouraged to erase their fears, trust their abilities and remove their baggage........and take the journey to empowerment and fulfilment in everything they do!"""},
                  {"title":"'Salem's Lot","description":"""#1 BESTSELLER â€¢ Ben Mears has returned to Jerusalemâ€™s Lot in hopes that exploring the history of the Marsten House, an old mansion long the subject of rumor and speculation, will help him cast out his personal devils and provide inspiration for his new book. But when two young boys venture into the woods, and only one returns alive, Mears begins to realize that something sinister is at work. In fact, his hometown is under siege from forces of darkness far beyond his imagination. And only he, with a small group of allies, can hope to contain the evil that is growing within the borders of this small New England town. With this, his second novel, Stephen King established himself as an indisputable master of American horror, able to transform the old conceits of the genre into something fresh and all the more frightening for taking place in a familiar, idyllic locale."""},
                  {"title":"It","description":"""Now with a stunning new cover look, King's classic No. 1 bestseller and the basis for the massively successful films It: Chapter One and It: Chapter Two as well as the inspiration for HBO Max's upcoming Welcome to Derry. We all float down here. Derry, Maine is just an ordinary town: familiar, well-ordered for the most part, a good place to live. It is a group of children who see - and feel - what makes Derry so horribly different. In the storm drains, in the sewers, IT lurks, taking on the shape of every nightmare, each one's deepest dread. Sometimes IT appears as an evil clown named Pennywise and sometimes IT reaches up, seizing, tearing, killing . . . Time passes and the children grow up, move away and forget. Until they are called back, once more to confront IT as IT stirs and coils in the sullen depths of their memories, emerging again to make their past nightmares a terrible present reality."""},
                  {"title":"Misery","description":"""After an almost fatal car crash, novelist Paul Sheldon finds himself being nursed by a deranged fan who holds him captive"""},
                  {"title":"Pet Sematary","description":"""""Sometimes dead is better...."" When the Creeds move into a beautiful old house in rural Maine, it all seems too good to be true: physician father, beautiful wife, charming little daughter, adorable infant son -- and now an idyllic home. As a family, they've got it all...right down to the friendly cat. But the nearby woods hide a blood-chilling truth -- more terrifying than death itself...and hideously more powerful."""},
                  {"title":"Fairy Tale","description":"""Paperback edition includes reading group guide."""},
                  {"title":"Cell","description":"""The latest, terrifying, #1 "New York Times" bestseller by Stephen King, about the mayhem unleashed when a mysterious force transforms cell phone users into homicidal maniacs, is available in a Premium Edition paperback."""},
                  {"title":"The Day of the Jackal","description":"""The Jackal. A tall, blond Englishman with opaque, gray eyes. A killer at the top of his profession. A man unknown to any secret service in the world. An assassin with a contract to kill the world's most heavily guarded man. One man with a rifle who can change the course of history. One man whose mission is so secretive not even his employers know his name. And as the minutes count down to the final act of execution, it seems that there is no power on earth that can stop the Jackal."""},
                  {"title":"The Dogs of War","description":"""In a remote corner of Zangaro, a small republic in Africa, lies Crystal Mountain. At certain times of the day the mountain emits a strange glow. Only Sir James Manson knows why. The mountain contains ten billion dollar's worth of the world's most valuable mineral, platinum. " Not only exciting but truly surprising" --Atlantic. Now the only question is, how to get hold of it. Sir James knows how. Invade the country with a band of savage, cold-blooded mercenaries. Topple the government and set up a puppet dictatorship. Unleash the dogs of war."""},
                  {"title":"The Mysterious Affair at Styles","description":"""The heiress of Styles has been murdered, dying in agony from strychnine slipped into her coffee. And there are plenty who would gain from her death: the financially strapped stepson, the gold digging younger husband, and an embittered daughter-in-law. Agatha Christie's eccentric and hugely popular detective, Hercule Poirot, was introduced to the world in this book, which launched her career as the most famous and best loved of all mystery writers."""},
                  {"title":"The Murder on the Links","description":"""The Murder on the Links is a work of detective fiction by Agatha Christie. The story takes place in northern France, giving Poirot a hostile competitor from the Paris SÃ»retÃ©. Poirot's long memory for past or similar crimes proves useful in resolving the crimes. The book is notable for a subplot in which Hastings falls in love, a development "greatly desired on Agatha's part... parcelling off Hastings to wedded bliss in the Argentine." Hercule Poirot and Captain Hastings travel to Merlinville-sur-Mer, France, to meet Paul Renauld, who has requested their help. Upon arriving at his home, the Villa Genevieve, local police greet them with news that he has been found dead that morning. Renauld had been stabbed in the back with a letter opener and left in a newly dug grave adjacent to a local golf course. His wife, Eloise Renauld, claims masked men broke into the villa at 2 am, tied her up, and took her husband away with them. Upon inspecting his body, Eloise collapses with grief at seeing her dead husband. Monsieur Giraud of the SÃ»retÃ© leads the police investigation, and resents Poirot's involvement; Monsieur Hautet, the Examining Magistrate, is more open to sharing key information with him. Poirot notes four key facts about the case: a piece of lead piping is found near the body; only three female servants were in the villa as both Renauld's son Jack and his chauffeur had been sent away; an unknown person visited the day before, whom Renauld urged to leave immediately; Renauld's immediate neighbour, Madame Daubreuil, had placed 200,000 francs into her bank account over recent weeks. When Renauld's secretary, Gabriel Stonor, returns from England, he suggests blackmail, as his employer's past is a complete mystery prior to his career in South America. Meanwhile, Hastings unexpectedly encounters a young woman he met before, known to him as "Cinderella", who asks to see the crime scene, and then mysteriously disappears with the murder weapon. Poirot later travels to Paris to research the case's similarities to that of a murder case from 22 years ago, which has only one difference - the killer, Georges Conneau, later confessed to the crime, in which he and his lover, Madame Beroldy, had plotted to kill her husband and claim that the murder was carried out by masked intruders; both disappeared soon afterwards. Returning from Paris, Poirot learns that the body of an unknown man has been found, stabbed through the heart with the murder weapon. An examination shows he has the hands of a tramp, that he died before Renauld's murder from an epileptic fit, and that he was stabbed after death. Giraud arrests Jack on the basis he wanted his father's money; Jack had admitted to police he had argued with his father over wishing to marry Mme Daubreuil's daughter Marthe, whom his parents found unsuitable. Poirot reveals a flaw in Giraud's theory, as Renauld changed his will two weeks before his murder, disinheriting Jack. Soon afterwards, Jack is released from prison after Bella Duveen, an English stage performer he loves, confesses to the murder. Both had come across the body on the night of the murder, and assumed the other had killed Renauld. Poirot reveals neither did, as the real killer was Marthe Daubreuil."""},
                  {"title":"Agatha Christie","description":"""The Queen of Crime's web site includes biographical information, books & plays, and links to tv and movies based on her characters. Follow the Tuesday Club Murders where a group of friends gather by the hearth and exchange intriguing tales of true crime."""},
                  {"title":"Poirot Investigates","description":"""Poirot Investigates a host of murders most foulâ€”as well as other dastardly crimesâ€”in this intriguing collection of short stories from the one-and-only Agatha Christie. First there was the mystery of the film star and the diamond . . . then came the â€œsuicideâ€ that was murder . . . the mystery of the absurdly cheap flat . . .a suspicious death in a locked gun room . . . a million dollar bond robbery . . . the curse of a pharaohâ€™s tomb . . . a jewel robbery by the sea . . . the abduction of a prime minister . . . the disappearance of a banker . . . a phone call from a dying man . . .and, finally, the mystery of the missing will. What links these fascinating cases? Only the brilliant deductive powers of Hercule Poirot!"""},
                  {"title":"Five Little Pigs","description":"""Amyas Crale was a celebrated painter . . . and an even more celebrated lover. His wife Caroline was as jealous as she was devoted. So naturally, she was convicted of Amyas' murder. Now, 16 years later, their daughter presents Poirot with a challenge: find the fatal flaw in the case that will clear her mother's name. Also published as Murder in Retrospect. Copyright Â© Libri GmbH. All rights reserved."""},
                  {"title":"The Big Four","description":"""In Agatha Christie's thrilling mystery novel, "The Big Four," readers are introduced to a cunning ensemble of criminal masterminds whose aim is global domination. Written in Christie's signature style of intricate plotting and well-developed characters, the narrative is woven with a blend of suspense and wit, typical of the detective fiction genre from the 1920s. Set against the backdrop of international intrigue, Poirot and Hastings take on an enigmatic challenge that stretches the limits of their deductive abilities, delving into the machinations of the titular quartet. The episodic structure of the novel, along with its emphasis on psychological tension, offers a refreshing departure from standard whodunits, placing it within the larger context of interwar literature that often wrestles with themes of power and corruption. Agatha Christie, often dubbed the "Queen of Crime," was notably influenced by her experiences serving as a pharmacy assistant during World War I, where she developed an understanding of poisons and human behavior. This knowledge permeates her work, contributing to the meticulous construction of her plots and characters. Additionally, her travels and interactions with diverse cultures provided a richly textured backdrop for her storytelling, further enhancing the complex dynamics of her narratives. For fans of mystery and adventure, "The Big Four" is a must-read that not only showcases Christie's unrivaled craftsmanship but also invites readers to engage with the intellectual challenge of unraveling a labyrinthine plot. With its blend of excitement, intellectual stimulation, and human psychology, this novel stands as a testament to Christie's legacy in the world of detective fiction."""},
                  {"title":"Murder on the Orient Express","description":"""Murder on the Orient Express is a detective novel by English writer Agatha Christie featuring the Belgian detective Hercule Poirot. It was first published in the United Kingdom by the Collins Crime Club on 1 January 1934. In the United States, it was published on 28 February 1934, under the title of Murder in the Calais Coach, by Dodd, Mead and Company. The elegant train of the 1930s, the Orient Express, is stopped by heavy snowfall. A murder is discovered, and Poirot's trip home to London from the Middle East is interrupted to solve the murder."""},
                  {"title":"Resurrection Walk","description":"""From #1 bestselling author Michael Connelly: Lincoln Lawyer Mickey Haller enlists the help of his half-brother, Harry Bosch, to prove the innocence of a woman convicted of killing her husband."""},
                  {"title":"Pines","description":""""Secret service agent Ethan Burke arrives in Wayward Pines, Idaho, with a clear mission: locate and recover two federal agents who went missing in the bucolic town one month earlier. But within minutes of his arrival, Ethan is involved in a violent accident. He comes to in a hospital, with no ID, no cell phone, and no briefcase. The medical staff seems friendly enough, but something feels ... off. As the days pass, Ethan's investigation into the disappearance of his colleagues turns up more questions than answers. Why can't he get any phone calls through to his wife and son in the outside world? Why doesn't anyone believe he is who he says he is? And what is the purpose of the electrified fences surrounding the town? Are they meant to keep the residents in? Or something else out? Each step closer to the truth takes Ethan further from the world he thought he knew, from the man he thought he was, until he must face a horrifying fact - he may never get out of Wayward Pines alive" -- Author's website."""},
                  {"title":"Abandon","description":"""A century-old mystery - and a desperate battle to survive. Abandon is a compulsive standalone thriller from Blake Crouch, the New York Times bestselling author of Dark Matter and Recursion. A century-old mystery, and a desperate battle to survive . . . On Christmas Day in 1893, every man, woman, and child in a remote mining town disappeared, belongings forsaken, meals left to freeze in vacant cabins, and not a single bone found. Now, journalist Abigail Foster and her historian father have set out to explore the long-abandoned town and learn what happened. With them are two backcountry guides along with a psychic and a paranormal photographer who are there to investigate rumours that the town is haunted. But Abigail and her companions are about to learn that the townâ€™s ghosts are the least of their worries. Twenty miles from civilization, with a blizzard bearing down, they realize they are not alone. The ordeal that follows will test this small team past the breaking point as they battle the elements and human foes alike and discover that the townâ€™s secrets still have the power to kill. Part journey into old-West history, part nail-biting survival thriller, Abandon is a bloody, darkly surprising tale as only Blake Crouch could deliver."""},
                  {"title":"Recursion","description":"""NEW YORK TIMES BESTSELLER â€¢ From the bestselling author of Dark Matter and the Wayward Pines trilogy comes a relentless thriller about time, identity, and memoryâ€”his most mind-boggling, irresistible work to date, and the inspiration for Shondalandâ€™s upcoming Netflix film. â€œGloriously twisting . . . a heady campfire tale of a novel.â€â€”The New York Times Book Review NAMED ONE OF THE BEST BOOKS OF THE YEAR BY Time â€¢ NPR â€¢ BookRiot Reality is broken. At first, it looks like a disease. An epidemic that spreads through no known means, driving its victims mad with memories of a life they never lived. But the force thatâ€™s sweeping the world is no pathogen. Itâ€™s just the first shock wave, unleashed by a stunning discoveryâ€”and whatâ€™s in jeopardy is not our minds but the very fabric of time itself. In New York City, Detective Barry Sutton is closing in on the truthâ€”and in a remote laboratory, neuroscientist Helena Smith is unaware that she alone holds the key to this mystery . . . and the tools for fighting back. Together, Barry and Helena will have to confront their enemyâ€”before they, and the world, are trapped in a loop of ever-growing chaos. Praise for Recursion â€œAn action-packed, brilliantly unique ride that had me up late and shirking responsibilities until I had devoured the last page . . . a fantastic read.â€â€”Andy Weir, #1 New York Times bestselling author of The Martian â€œAnother profound science-fiction thriller. Crouch masterfully blends science and intrigue into the experience of what it means to be deeply human.â€â€”Newsweek â€œDefinitely not one to forget when youâ€™re packing for vacation . . . [Crouch] breathes fresh life into matters with a mix of heart, intelligence, and philosophical musings.â€â€”Entertainment Weekly â€œA trippy journey down memory lane . . . [Crouchâ€™s] intelligence is an able match for the challenge heâ€™s set of overcoming the structure of time itself.â€â€”Time â€œWildly entertaining . . . another winning novel from an author at the top of his game.â€â€”AV Club"""},
                  {"title":"Dark Matter","description":"""A mind-bending, relentlessly paced science-fiction thriller, in which an ordinary man is kidnapped, knocked unconscious--and awakens in a world inexplicably different from the reality he thought he knew."""}]
    cleaned_df = novel_df.copy()

    for entry in target_entries:
        target_title = entry["title"]
        target_description = entry["description"]

        title_mask = cleaned_df["title"].str.lower() == target_title.lower()

        description_mask = cleaned_df["description"] == target_description

        keep_mask = title_mask & description_mask

        cleaned_df = cleaned_df[~title_mask | keep_mask]

    return cleaned_df



def drop_titles(df):
    """
    Drops all rows where the 'title' column matches any of these specified duplicated titles/titles that'll introduce a bias into our model 
    because they're part of a series already (making sure they're case-insensitive).
    
    Parameters:
    - df: DataFrame with a 'title' column
    - titles_to_remove: list of titles to be dropped

    Returns:
    - A filtered DataFrame
    """
    titles_to_remove = [
    "The Black Ice",
    "The Last Coyote",
    "Trunk Music",
    "A Darkness More Than Night",
    "City of Bones",
    "The Wrong Side of Goodbye",
    "Two Kinds of Truth",
    "Resurrection Walk",
    "The Reversal",
    "The Fifth Witness",
    "Wayward",
    "The Last Town"
]
    # Normalize title column and the list to lowercase for comparison
    titles_to_remove_lower = [title.lower() for title in titles_to_remove]
    return df[~df['title'].str.lower().isin(titles_to_remove_lower)]


def add_author_stats_new(df):
    """
    Adds author-level Goodreads statistics to df.
    
    Parameters:
    - df: DataFrame with 'author_searched'
    - author_stats: dict mapping author name -> dict with keys
        'avg_rating', 'rating_count', 'total_reviews'
    
    Returns:
    - DataFrame augmented with three new columns:
      'author_avg_rating', 'author_rating_count', 'author_total_reviews'
    """
    author_stats = {
    "Dan Brown": {
        "avg_rating": 3.90,
        "rating_count": 8851474,
        "total_reviews": 219115
    },
    "Chinua Achebe": {
        "avg_rating": 3.76,
        "rating_count": 452578,
        "total_reviews": 27163
    },
    "Chimamanda Ngozi Adichie":{
         "avg_rating": 4.31,
        "rating_count": 1295770,
        "total_reviews": 119314
    },
    "George Orwell":{
         "avg_rating": 4.11,
        "rating_count": 10253035,
        "total_reviews": 305087
    },
    "John Grisham":{
         "avg_rating": 3.97,
        "rating_count": 6712177,
        "total_reviews": 266168
    },
    "Stephen King":{
         "avg_rating": 4.07,
        "rating_count": 21440043,
        "total_reviews": 1080988
    },
    "Frederick Forsyth":{
        "avg_rating": 4.1,
        "rating_count": 420205,
        "total_reviews": 13368
    },
    "Agatha Christie":{
      "avg_rating": 4.02,
        "rating_count": 7778022,
        "total_reviews": 511814  
    },
    "Michael Connelly":{
        "avg_rating": 4.21,
        "rating_count": 3219028,
        "total_reviews": 159493  
    },
    "John Green":{
        "avg_rating": 3.98,
        "rating_count": 10789515,
        "total_reviews": 491869
    },
    "Blake Crouch":{
         "avg_rating": 4.06,
        "rating_count": 1568698,
        "total_reviews": 172722
    },
    "Sidney Sheldon":{
         "avg_rating": 3.88,
        "rating_count": 678654,
        "total_reviews": 25137
    },
    "Mick Herron":{
        "avg_rating": 4.18,
        "rating_count": 336349,
        "total_reviews": 25268
    },
    "Rachel Howzell Hall":{
        "avg_rating": 3.57,
        "rating_count": 93141,
        "total_reviews": 9609
    },
    "Kristin Hannah":{
         "avg_rating": 4.4,
        "rating_count": 7767962,
        "total_reviews": 665021
    }
}
    df = df.copy()
    df['author_searched_norm'] = df['author_searched'].str.strip().str.lower()
    author_stats_norm = {k.strip().lower(): v for k, v in author_stats.items()}
    df['author_avg_rating'] = df['author_searched_norm'].map(
        lambda a: author_stats_norm.get(a, {}).get('avg_rating', None)
    )
    df['author_rating_count'] = df['author_searched_norm'].map(
        lambda a: author_stats_norm.get(a, {}).get('rating_count', None)
    )
    df['author_total_reviews'] = df['author_searched_norm'].map(
        lambda a: author_stats_norm.get(a, {}).get('total_reviews', None)
    )
    return df.drop(columns=['author_searched_norm'])



def filter_author_titles(df, author, valid_titles):
    """
    Keeps only rows where the author matches and the title is in the valid list.
    All other rows for that author are removed because they are mostly books belonging to a series, and that can introduce a bias into our model.
    All other authors' rows are preserved.

    Parameters:
    - df: DataFrame with 'author_searched' and 'title' columns
    - author: string name of the author to filter
    - valid_titles: list of valid titles for that author

    Returns:
    - A filtered DataFrame
    """
    valid_titles_clean = [title.strip() for title in valid_titles]
    mask = (df['author_searched'] != author) | (df['title'].isin(valid_titles_clean))
    return df[mask]


valid_blake_crouch_titles = [
    "Dark Matter",
    "Upgrade",
    "Pines",
    "Abandon",
    "Recursion"
]

valid_john_green_titles = [
    "Looking for Alaska",
    "An Abundance of Katherines",
    "The Fault in Our Stars",
    "Paper Towns",
    "Will Grayson Will Grayson",
    "Turtles All the Way Down",
    "Let It Snow",
    "Everything Is Tuberculosis"
]

valid_sidney_sheldon_titles = [
    "IF TOMORROW COMES",
    "Master of the Game",
    "Bloodline",
    "The Other Side of Midnight",
    "Tell Me Your Dreams",
    "The Sands of Time",
    "The Sky is Falling",
    "A Stranger in the Mirror"
]

valid_mick_herron_titles = [
    "Slow Horses",
    "Nobody Walks",
    "The Secret Hours",
    "This Is What Happened",
    "Reconstruction"
]

valid_rachel_hall_titles = [
    "These Toxic Things",
    "What Never Happened",
    "We Lie Here",
    "They All Fall Down",
    "And Now She's Gone",
    "A Quiet Storm"
]

valid_kristin_hannah_titles = [
    "The Nightingale",
    "The Women",
    "Magic Hour",
    "Angel Falls",
    "When Lightning Strikes",
    "The Enchantment",
    "The Great Alone",
    "Home Front"
]

def apply_title_filter(df):
    author_title_map = {
        'Blake Crouch': valid_blake_crouch_titles,
        'John Green': valid_john_green_titles,
        'Sidney Sheldon': valid_sidney_sheldon_titles,
        'Rachel Howzell Hall': valid_rachel_hall_titles,
        'Mick Herron': valid_mick_herron_titles,
        'Kristin Hannah': valid_kristin_hannah_titles
    }

    for author, valid_titles in author_title_map.items():
        df = filter_author_titles(df, author, valid_titles)
    
    return df


import requests
from bs4 import BeautifulSoup
import time
import random

# Retry-enabled Goodreads scraper
def scrape_goodreads_with_retry(title, author=None, max_retries=3, sleep_range=(1, 3)):
    query = f"{title} {author}" if author else title
    search_url = f"https://www.goodreads.com/search?q={requests.utils.quote(query)}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code != 200:
                raise Exception("Non-200 response")

            soup = BeautifulSoup(response.content, "html.parser")
            book_link = soup.select_one("a.bookTitle")
            if not book_link:
                raise Exception("Book link not found")

            book_url = "https://www.goodreads.com" + book_link["href"]
            book_response = requests.get(book_url, headers={"User-Agent": "Mozilla/5.0"})
            if book_response.status_code != 200:
                raise Exception("Failed to load book page")

            book_soup = BeautifulSoup(book_response.content, "html.parser")

            avg_rating = book_soup.select_one("span[itemprop='ratingValue']")
            rating_count = book_soup.select_one("meta[itemprop='ratingCount']")
            review_count = book_soup.select_one("meta[itemprop='reviewCount']")
            num_pages = book_soup.select_one("span[itemprop='numberOfPages']")
            pub_year = book_soup.select_one("nobr.greyText")

            return {
                "author_avg_rating": float(avg_rating.text.strip()) if avg_rating else None,
                "author_rating_count": int(rating_count["content"].replace(',', '')) if rating_count else None,
                "author_total_reviews": int(review_count["content"].replace(',', '')) if review_count else None,
                "num_pages": int(num_pages.text.strip().split()[0]) if num_pages else None,
                "published_year": int(pub_year.text.strip()[-5:-1]) if pub_year else None
            }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(random.uniform(*sleep_range))
            else:
                print(f"Retry failed for '{title}': {str(e)}")
                return None

# Hybrid fallback to Google Books API
import requests

import requests
from difflib import get_close_matches

import requests
from difflib import get_close_matches

import requests

def fetch_from_google_books(title, author=None, api_key=None):
    print(f"🔍 Searching for: '{title}' by '{author}'")

    try:
        query = f"{title} {author}" if author else title
        url = f"https://www.googleapis.com/books/v1/volumes?q={requests.utils.quote(query)}"
        if api_key:
            url += f"&key={api_key}"

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "items" not in data:
            print("⚠️ No items found.")
            return None

        for item in data["items"]:
            info = item.get("volumeInfo", {})
            avg_rating = info.get("averageRating")
            ratings_count = info.get("ratingsCount")

            if avg_rating is not None and ratings_count is not None:
                print(f"✅ Match: {info.get('title')} | Rating: {avg_rating}, Count: {ratings_count}")
                return {
                    "author_avg_rating": avg_rating,
                    "author_rating_count": ratings_count,
                    "author_total_reviews": None,  
                    "num_pages": info.get("pageCount"),
                    "published_year": int(info["publishedDate"][:4]) if "publishedDate" in info else None
                }

        print("❌ No item with ratings found.")
        return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


from tqdm import tqdm

tqdm.pandas()

GOOGLE_BOOKS_API_KEY = "AIzaSyC6Dk5o102Kxw7MgpuUBpgvQDq16G91z5o"

def enrich_row(row, fetch_func):
    """
Adds book-level stats to enrich every row of my dataframe.Stats like the book_avg_rating,num_pages etc. 
    """
    title = row["title"]
    author = row["author_searched"]
    data = fetch_func(title, author, api_key=GOOGLE_BOOKS_API_KEY)

    if data:
        row["book_avg_rating"] = data.get("author_avg_rating")
        row["book_rating_count"] = data.get("author_rating_count")
        row["book_total_reviews"] = data.get("author_total_reviews")
        row["num_pages"] = data.get("num_pages")
        row["book_published_year"] = data.get("published_year")
    
    return row

def enrich_novel_df(novel_df, fetch_func):
    # Add empty columns
    novel_df["book_avg_rating"] = None
    novel_df["book_rating_count"] = None
    novel_df["book_total_reviews"] = None
    novel_df["num_pages"] = None
    novel_df["book_published_year"] = None

    return novel_df.apply(lambda row: enrich_row(row, fetch_func), axis=1)

def drop_these_columns(df):
    useless_columns = ['average_rating','ratings_count','published_date']
    df = df.drop(columns=useless_columns)
    return df 

def drop_empty_rows(df,col_name):
    return df.dropna(subset=[col_name])


def clean_novel_data(df):
    df = keep_only_correct_variants(df)
    df = drop_titles(df)
    df = add_author_stats_new(df)
    df = apply_title_filter(df)
    df = enrich_novel_df(df,fetch_from_google_books)
    df = drop_these_columns(df)
    df = drop_empty_rows(df,'description')
    return df

def some_more_cleaning(df):
    from sklearn.impute import SimpleImputer
    import pandas as pd

    imputer = SimpleImputer(strategy='mean')

    df.drop(columns=['book_total_reviews', 'book_rating_count', 'authors'], inplace=True)

    df['categories'] = df['categories'].astype(str).str.strip('[').str.strip(']')
    df['categories'] = df['categories'].fillna(method='ffill')

    num_pages_imputed = imputer.fit_transform(df[['num_pages']])
    df['num_pages'] = pd.Series(num_pages_imputed.flatten(), index=df.index).astype(int)

    published_year_imputed = imputer.fit_transform(df[['book_published_year']])
    df['book_published_year'] = pd.Series(published_year_imputed.flatten(), index=df.index).astype(int)

    avg_rating_imputed = imputer.fit_transform(df[['book_avg_rating']])
    df['book_avg_rating'] = pd.Series(avg_rating_imputed.flatten(), index=df.index).round(1)

    return df


def drop_duplicate_rows(df):
    cols_to_check = [
        'author_avg_rating',
        'author_rating_count',
        'author_total_reviews',
        'book_avg_rating',
        'num_pages',
        'book_published_year'
    ]
    
    # Drop duplicates based on these columns
    df = df.drop_duplicates(subset=cols_to_check, keep=False)
    
    return df


def column_conversions(df):
    import pandas as pd 
    for col in df.columns:
        if col in ['categories', 'author_searched', 'title']:
            df[col] = df[col].astype('category')
        elif col in ['author_avg_rating', 'book_avg_rating']:
            df[col] = df[col].astype('float32') 
        elif col in ['author_rating_count', 'author_total_reviews', 'num_pages', 'book_published_year']:
            df[col] = df[col].astype('int32')
    df.to_csv('clean_1.csv', index=False)
    return df



def more_rows_and_inference_data(filename):
    books_data = [
    {'title': 'The Da Vinci Code (The Young Adult Adaptation)', 'book_avg_rating': 3.93, 'book_ratings': 2479509, 'book_reviews': 58517},
    {'title': 'The Lost Symbol', 'book_avg_rating': 3.76, 'book_ratings': 641330, 'book_reviews': 30920},
    {'title': 'The Lost Symbol Illustrated edition', 'book_avg_rating': 3.66, 'book_ratings': 623204, 'book_reviews': 20423},
    {'title': 'Wild Symphony', 'book_avg_rating': 4.3, 'book_ratings': 1456, 'book_reviews': 316},
    {'title': 'Digital Fortress', 'book_avg_rating': 3.70, 'book_ratings': 658262, 'book_reviews': 11631},
    {'title': 'Dan Brown’s Robert Langdon Series', 'book_avg_rating': 3.8, 'book_ratings': 540223, 'book_reviews': 43442},
    {'title': 'In One Sitting', 'book_avg_rating': 5.0, 'book_ratings': 1, 'book_reviews': 0},
    {'title': 'The Lost Symbol: Special Illustrated Edition', 'book_avg_rating': 3.66, 'book_ratings': 623204, 'book_reviews': 20423},
    {'title': 'Supergirl Vol. 6: Crucible', 'book_avg_rating': 2.5, 'book_ratings': 2332, 'book_reviews': 321},
    {'title': 'The Trouble with Nigeria', 'book_avg_rating': 4.13, 'book_ratings': 313, 'book_reviews': 44},
    {'title': 'No Longer at Ease', 'book_avg_rating': 3.87, 'book_ratings': 12627, 'book_reviews': 1086},
    {'title': 'A Man of the People', 'book_avg_rating': 3.89, 'book_ratings': 5447, 'book_reviews': 458},
    {'title': 'There Was a Country', 'book_avg_rating': 4.03, 'book_ratings': 2741, 'book_reviews': 362},
    {'title': 'Another Africa', 'book_avg_rating': 3.71, 'book_ratings': 24, 'book_reviews': 3},
    {'title': 'Girls at War', 'book_avg_rating': 3.74, 'book_ratings': 1570, 'book_reviews': 167},
    {'title': 'Arrow of God', 'book_avg_rating': 3.79, 'book_ratings': 8965, 'book_reviews': 774},
    {'title': 'Collected Poems', 'book_avg_rating': 3.73, 'book_ratings': 407, 'book_reviews': 49},
    {'title': 'Things Fall Apart, Chinua Achebe', 'book_avg_rating': 3.74, 'book_ratings': 403910, 'book_reviews': 22500},
    {'title': 'Things Fall Apart SparkNotes Literature Guide', 'book_avg_rating': 4.0, 'book_ratings': 1, 'book_reviews': 0},
    {'title': 'Purple Hibiscus', 'book_avg_rating': 4.18, 'book_ratings': 137732, 'book_reviews': 11557},
    {'title': 'We Should All Be Feminists: A Guided Journal', 'book_avg_rating': 4.40, 'book_ratings': 321546, 'book_reviews': 31258},
    {'title': 'One World', 'book_avg_rating': 4.25, 'book_ratings': 4, 'book_reviews': 0},
    {'title': 'Freedom', 'book_avg_rating': 3.6, 'book_ratings': 432, 'book_reviews': 32},
    {'title': 'Why I Write', 'book_avg_rating': 3.1, 'book_ratings': 322, 'book_reviews': 10},
    {'title': 'ANIMAL FARM《动物庄园》', 'book_avg_rating': 4.01, 'book_ratings': 4374066, 'book_reviews': 123068},
    {'title': "Orwell's Nineteen Eighty-four", 'book_avg_rating': 4.20, 'book_ratings': 5247668, 'book_reviews': 144758},
    {'title': 'Animal Farm a Fairy Story', 'book_avg_rating': 4.01, 'book_ratings': 4365045, 'book_reviews': 123035},
    {'title': 'The Collected Essays, Journalism and Letters of George Orwell: An age like this, 1920-1940', 'book_avg_rating': 2.8, 'book_ratings': 43, 'book_reviews': 54},
    {'title': 'Nineteen Eighty Four Anniversary Edition', 'book_avg_rating': 4.1, 'book_ratings': 407465, 'book_reviews': 3214},
    {'title': 'Animal Farm: a Fairy Story (Illustrated)', 'book_avg_rating': 3.8, 'book_ratings': 432978, 'book_reviews': 23866},
    {'title': 'Facing Unpleasant Facts', 'book_avg_rating': 4.50, 'book_ratings': 22, 'book_reviews': 5},
    {'title': '1984 by George Orwell', 'book_avg_rating': 4.3, 'book_ratings': 4343214, 'book_reviews': 208774},
    {'title': 'Modern Classics Nineteen Eighty Four', 'book_avg_rating': 4.20, 'book_ratings': 5247668, 'book_reviews': 144758},
    {'title': 'Down and Out in Paris and London', 'book_avg_rating': 4.09, 'book_ratings': 97858, 'book_reviews': 7291},
    {'title': 'The Rainmaker', 'book_avg_rating': 4.01, 'book_ratings': 204375, 'book_reviews': 2920},
    {'title': 'A Time for Mercy', 'book_avg_rating': 4.29, 'book_ratings': 95128, 'book_reviews': 7067},
    {'title': 'Sycamore Row', 'book_avg_rating': 4.10, 'book_ratings': 137928, 'book_reviews': 11476},
    {'title': 'A Painted House', 'book_avg_rating': 3.77, 'book_ratings': 100572, 'book_reviews': 5325},
    {'title': 'The Street Lawyer', 'book_avg_rating': 3.89, 'book_ratings': 121096, 'book_reviews': 3334},
    {'title': 'The Testament', 'book_avg_rating': 3.91, 'book_ratings': 124760, 'book_reviews': 4049},
    {'title': 'The Runaway Jury', 'book_avg_rating': 4.03, 'book_ratings': 312436, 'book_reviews': 2981},
    {'title': 'Rogue Lawyer', 'book_avg_rating': 3.80, 'book_ratings': 87098, 'book_reviews': 7118},
    {'title': 'Ford County: Stories', 'book_avg_rating': 3.72, 'book_ratings': 28090, 'book_reviews': 2513},
    {'title': 'Camino Island', 'book_avg_rating': 3.82, 'book_ratings': 158279, 'book_reviews': 12130},
    {'title': 'Camino Winds', 'book_avg_rating': 3.81, 'book_ratings': 89133, 'book_reviews': 6316},
    {'title': 'The Body', 'book_avg_rating': 4.02, 'book_ratings': 33416, 'book_reviews': 3610},
    {'title': 'Fairy Tale', 'book_avg_rating': 4.09, 'book_ratings': 321933, 'book_reviews': 36409},
    {'title': 'Carrie', 'book_avg_rating': 3.99, 'book_ratings': 815961, 'book_reviews': 31533},
    {'title': 'The Dark Tower I', 'book_avg_rating': 3.91, 'book_ratings': 651329, 'book_reviews': 27802},
    {'title': 'Bag of Bones', 'book_avg_rating': 3.92, 'book_ratings': 208491, 'book_reviews': 6501},
    {'title': 'Joyland', 'book_avg_rating': 3.93, 'book_ratings': 174151, 'book_reviews': 17038},
    {'title': 'Four Past Midnight', 'book_avg_rating': 3.95, 'book_ratings': 114684, 'book_reviews': 2574},
    {'title': 'The Tommyknockers', 'book_avg_rating': 3.60, 'book_ratings': 162168, 'book_reviews': 4538},
    {'title': 'Rose Madder', 'book_avg_rating': 3.76, 'book_ratings': 121587, 'book_reviews': 4536},
    {'title': 'Sleeping Beauties', 'book_avg_rating': 3.73, 'book_ratings': 92405, 'book_reviews': 9656},
    {'title': 'The Breathing Method', 'book_avg_rating': 3.64, 'book_ratings': 3331, 'book_reviews': 294},
    {'title': 'It. Film Tie-In', 'book_avg_rating': 4.24, 'book_ratings': 1230528, 'book_reviews': 40977},
    {'title': 'Just After Sunset', 'book_avg_rating': 3.88, 'book_ratings': 59431, 'book_reviews': 3586},
    {'title': 'It', 'book_avg_rating': 4.24, 'book_ratings': 1230528, 'book_reviews': 40977},
    {'title': 'The Drawing of the Three', 'book_avg_rating': 4.24, 'book_ratings': 283520, 'book_reviews': 11659},
    {'title': 'The Gunslinger', 'book_avg_rating': 3.91, 'book_ratings': 651329, 'book_reviews': 27802},
    {'title': 'Insomnia', 'book_avg_rating': 3.84, 'book_ratings': 165335, 'book_reviews': 5536},
    {"title": "Lisey's Story", 'book_avg_rating': 3.70, 'book_ratings': 92901, 'book_reviews': 6163},
    {'title': 'Thinner', 'book_avg_rating': 3.78, 'book_ratings': 223173, 'book_reviews': 4662},
    {"title": "Stephen King's Danse Macabre", 'book_avg_rating': 3.67, 'book_ratings': 30645, 'book_reviews': 1466},
    {'title': 'The Dark Tower III', 'book_avg_rating': 4.46, 'book_ratings': 8572, 'book_reviews': 119},
    {'title': 'If It Bleeds', 'book_avg_rating': 3.98, 'book_ratings': 119053, 'book_reviews': 12393},
    {'title': 'A Good Marriage', 'book_avg_rating': 3.83, 'book_ratings': 22967, 'book_reviews': 1696},
    {'title': 'Just After Sunset EXP', 'book_avg_rating': 3.88, 'book_ratings': 59431, 'book_reviews': 3586},
    {'title': 'Duma Key', 'book_avg_rating': 3.99, 'book_ratings': 133780, 'book_reviews': 7930},
    {'title': 'The Green Mile', 'book_avg_rating': 4.49, 'book_ratings': 352455, 'book_reviews': 15377},
    {'title': 'The Langoliers', 'book_avg_rating': 3.93, 'book_ratings': 43963, 'book_reviews': 1063},
    {'title': 'The Mist', 'book_avg_rating': 3.93, 'book_ratings': 174770, 'book_reviews': 7486},
    {'title': 'The Negotiator', 'book_avg_rating': 4.03, 'book_ratings': 13251, 'book_reviews': 324},
    {'title': 'The Biafra Story', 'book_avg_rating': 3.94, 'book_ratings': 583, 'book_reviews': 47},
    {'title': 'The Fist of God', 'book_avg_rating': 4.07, 'book_ratings': 13169, 'book_reviews': 383},
    {'title': 'The Outsider', 'book_avg_rating': 4.16, 'book_ratings': 2526, 'book_reviews': 310},
    {'title': 'Avenger', 'book_avg_rating': 4.08, 'book_ratings': 12088, 'book_reviews': 596},
    {'title': 'The Phantom of Manhattan', 'book_avg_rating': 3.02, 'book_ratings': 3511, 'book_reviews': 416},
    {'title': 'The Dogs of War', 'book_avg_rating': 3.99, 'book_ratings': 24808, 'book_reviews': 601},
    {'title': 'Cards on the Table', 'book_avg_rating': 3.96, 'book_ratings': 69478, 'book_reviews': 4802},
    {"title": "Agatha Christie's Murder on the Orient Express", 'book_avg_rating': 4.20, 'book_ratings': 705161, 'book_reviews': 46673},
    {'title': 'Hickory Dickory Dock', 'book_avg_rating': 3.76, 'book_ratings': 37592, 'book_reviews': 2407},
    {'title': 'Murder is Easy', 'book_avg_rating': 3.77, 'book_ratings': 31342, 'book_reviews': 2601},
    {'title': 'The ABC Murders', 'book_avg_rating': 4.03, 'book_ratings': 191436, 'book_reviews': 12320},
    {'title': 'Death in the Clouds', 'book_avg_rating': 3.86, 'book_ratings': 63286, 'book_reviews': 4208},
    {'title': 'Death on the Nile', 'book_avg_rating': 4.12, 'book_ratings': 295306, 'book_reviews': 20622},
    {'title': 'A Christmas Tragedy', 'book_avg_rating': 3.82, 'book_ratings': 2610, 'book_reviews': 197},
    {'title': 'And Then There Were None', 'book_avg_rating': 4.28, 'book_ratings': 1550339, 'book_reviews': 77704},
    {'title': 'The Sittaford Mystery', 'book_avg_rating': 3.77, 'book_ratings': 28926, 'book_reviews': 2389},
    {'title': 'A Caribbean Mystery', 'book_avg_rating': 3.17, 'book_ratings': 6, 'book_reviews': 1},
    {'title': 'A Murder is Announced', 'book_avg_rating': 4.01, 'book_ratings': 82624, 'book_reviews': 5574},
    {'title': 'The Man in the Brown Suit Agatha Christie', 'book_avg_rating': 4.5, 'book_ratings': 416, 'book_reviews': 14},
    {'title': 'Peril at End House', 'book_avg_rating': 4.02, 'book_ratings': 79791, 'book_reviews': 5636},
    {'title': 'Endless Night', 'book_avg_rating': 3.81, 'book_ratings': 42867, 'book_reviews': 4618},
    {'title': 'They Do it with Mirrors', 'book_avg_rating': 3.78, 'book_ratings': 44058, 'book_reviews': 2923},
    {'title': 'The Black Echo', 'book_avg_rating': 4.14, 'book_ratings': 212061, 'book_reviews': 6628},
    {'title': 'Angels Flight', 'book_avg_rating': 4.23, 'book_ratings': 67197, 'book_reviews': 2727},
    {'title': 'The Law of Innocence', 'book_avg_rating': 4.35, 'book_ratings': 73819, 'book_reviews': 5389},
    {'title': 'Switchblade', 'book_avg_rating': 3.70, 'book_ratings': 11391, 'book_reviews': 672},
    {'title': 'Echo Park', 'book_avg_rating': 4.18, 'book_ratings': 66608, 'book_reviews': 2749},
    {'title': 'Crime Beat', 'book_avg_rating': 3.39, 'book_ratings': 6000, 'book_reviews': 480},
    {'title': 'The Lincoln Lawyer', 'book_avg_rating': 4.22, 'book_ratings': 250917, 'book_reviews': 7935},
    {'title': 'The Drop', 'book_avg_rating': 4.20, 'book_ratings': 72999, 'book_reviews': 3757},
    {'title': 'An Abundance of Katherines', 'book_avg_rating': 3.51, 'book_ratings': 545787, 'book_reviews': 28939},
    {'title': 'Looking for Alaska', 'book_avg_rating': 3.96, 'book_ratings': 1719159, 'book_reviews': 81185},
    {'title': 'Let It Snow', 'book_avg_rating': 3.66, 'book_ratings': 157050, 'book_reviews': 13728},
    {'title': 'Upgrade', 'book_avg_rating': 3.81, 'book_ratings': 115040, 'book_reviews': 14950},
    {'title': 'Dark Matter', 'book_avg_rating': 4.13, 'book_ratings': 669896, 'book_reviews': 79646},
    {'title': 'Pines', 'book_avg_rating': 3.93, 'book_ratings': 167300, 'book_reviews': 15139},
    {'title': 'Master of the Game', 'book_avg_rating': 4.15, 'book_ratings': 59647, 'book_reviews': 2361},
    {'title': 'The Sky is Falling', 'book_avg_rating': 3.62, 'book_ratings': 22711, 'book_reviews': 829},
    {'title': 'The Other Side of Midnight', 'book_avg_rating': 3.96, 'book_ratings': 49835, 'book_reviews': 1498},
    {'title': 'A Stranger in the Mirror', 'book_avg_rating': 3.67, 'book_ratings': 21856, 'book_reviews': 616},
    {'title': 'Tell Me Your Dreams', 'book_avg_rating': 3.99, 'book_ratings': 54975, 'book_reviews': 2472},
    {'title': 'This Is What Happened', 'book_avg_rating': 3.44, 'book_ratings': 5259, 'book_reviews': 604},
    {'title': 'Nobody Walks', 'book_avg_rating': 4.11, 'book_ratings': 8064, 'book_reviews': 575},
    {'title': 'These Toxic Things', 'book_avg_rating': 3.77, 'book_ratings': 17948, 'book_reviews': 1477},
    {'title': 'We Lie Here', 'book_avg_rating': 3.85, 'book_ratings': 8903, 'book_reviews': 785},
    {'title': 'What Never Happened', 'book_avg_rating': 3.63, 'book_ratings': 26008, 'book_reviews': 1743},
    {'title': 'A Quiet Storm', 'book_avg_rating': 3.71, 'book_ratings': 648, 'book_reviews': 56},
    {'title': 'Home Front', 'book_avg_rating': 4.26, 'book_ratings': 206635, 'book_reviews': 15556},
    {'title': 'The Nightingale', 'book_avg_rating': 4.64, 'book_ratings': 1915199, 'book_reviews': 162080},
    {'title': 'Magic Hour', 'book_avg_rating': 4.17, 'book_ratings': 204816, 'book_reviews': 13987},
    {'title': 'When Lightning Strikes', 'book_avg_rating': 3.46, 'book_ratings': 2707, 'book_reviews': 246}
]

    import pandas as pd 
    cleaned_df = pd.read_csv(filename)
    books_df = pd.DataFrame(books_data)

    if 'book_avg_rating' in cleaned_df.columns:
        cleaned_df.drop(columns='book_avg_rating', inplace=True)

    cleaned_df= cleaned_df.merge(
    books_df[['title', 'book_avg_rating', 'book_ratings', 'book_reviews']],
    on='title',
    how='left'  
)
    binding = pd.read_csv('just_lemme_see.csv')
    rows_to_append = [0,1,3,4,6,7,8,22,23,26,27,28,29,30,34,35,36,40,41,44,45,46,47,48,49,54,55,56,60,61,62]
    append_df = binding.iloc[rows_to_append,:]
    cleaned_df = pd.concat([cleaned_df,append_df])
    binding = binding.drop(append_df.index)
    binding.to_csv('novel_inference_data.csv',index=False)
    return cleaned_df



def setting_expectations(df):
    """
The columns from my cleaned dataframe must meet these expectations before they can then be sent to Hopswork's feature store for temporary storage
    """
    import great_expectations as gx
    context = gx.get_context()
    data_source = context.data_sources.add_pandas(name = 'my_pandas_datasource')
    data_asset = data_source.add_dataframe_asset(name='my_dataframe_asset')
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name='my_batch_definition')
    batch = batch_definition.get_batch(batch_parameters={'dataframe':df})

    #Defining my expectations
    row_count = gx.expectations.ExpectTableRowCountToEqual(value=158)
    column_count = gx.expectations.ExpectTableColumnCountToEqual(value=12)
    book_avg_median = gx.expectations.ExpectColumnMedianToBeBetween(column='book_avg_rating',min_value=3.7,max_value=4.1)
    for col in df.columns:
        no_nulls_allowed = gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
    suite = gx.ExpectationSuite(name='my_expectation_suite')
    
    expectations = [row_count,column_count,book_avg_median,no_nulls_allowed]
    for expectation in expectations:
        suite.add_expectation(expectation)
    context.suites.add(suite)

    validation_definition = gx.ValidationDefinition(name='my_validation_definition',data=batch_definition,suite=suite)
    context.validation_definitions.add(validation_definition)
    checkpoint = gx.Checkpoint(name='my_checkpoint',validation_definitions=[validation_definition])
    checkpoint_results = checkpoint.run(batch_parameters={'dataframe':df})
    print(checkpoint_results.success)


import pandas as pd
import numpy as np
import re

def extract_features(df):
    """
    Extracts numeric and text-based features from my description column.
    Text features include:
      - buzzword_count (this is a likely indicator as to how much hype a novel has)
      - description word count
      - number of sentences
      - adjective count (simple regex-based)
    """
    # --- TEXT FEATURES ---
    buzzwords = ['award', 'bestseller', 'classic', 'legendary', 'masterpiece', 
                 'epic', 'thrilling', 'captivating', 'page-turner', 'unforgettable']

    df['description_clean'] = df['description'].astype(str).str.lower()

    df['buzzword_count'] = df['description_clean'].apply(
        lambda x: sum(x.count(word) for word in buzzwords)
    )
    df['desc_len_words'] = df['description_clean'].apply(lambda x: len(x.split()))
    df['num_sentences'] = df['description_clean'].apply(lambda x: x.count('.') + x.count('!') + x.count('?'))
    df['num_adjectives'] = df['description_clean'].str.count(r'\b\w+(ly|ous|ive|able|ful|ish|ic|al|ant|ent)\b')

    df = df.drop(columns=['description_clean'])
    
    return df


def create_hit_miss(df, quantile=0.35):
    """
    Creates a hit/miss column based on a weighted popularity score.
    score = (book_avg_rating * 100) + log(1 + book_reviews)

    Parameters:
        df : pandas.DataFrame
        quantile : float
            The percentile cutoff for defining a "Hit". 
            Default = 0.35 means top 65% of books will be Hits.

    Returns:
        df : pandas.DataFrame with added 'score' and 'hit_miss'
    """
    # --- Compute score ---
    df['score'] = (df['book_avg_rating'] * 100) + np.log1p(df['book_reviews'])

    # --- Threshold by quantile ---
    threshold = df['score'].quantile(quantile)
    df['hit_miss'] = np.where(df['score'] >= threshold, 'Hit', 'Miss')
    df.drop(columns='score',inplace=True)
    return df


def send_to_feature_store(df):
    import hopsworks
    api_key = 'NhIODCOOM50hfmRh.f8KWceqZECE4l3Id7GAr47u5MAw9Wn28KWFRU9JNXFm1oVh4bdDtN5sRigsS9cD8'
    project = hopsworks.login(api_key_value=api_key)

    fs = project.get_feature_store()
    novel_data_fg = fs.get_or_create_feature_group(name = 'novel_data',
                                                version = 1,
                                                description = 'Cleaned Novel Data Prediction Dataset',
                                                primary_key = ['author_searched', 'title', 'categories', 'author_avg_rating',
       'author_rating_count', 'author_total_reviews', 'num_pages',
       'book_published_year', 'book_avg_rating', 'book_ratings',
       'book_reviews', 'buzzword_count', 'desc_len_words', 'num_sentences',
       'num_adjectives', 'hit_miss'],
       online_enabled=False)
    novel_data_fg.insert(df, wait=True)
    commits=novel_data_fg.commit_details()
    print(f'commits: {commits}')

def main():
    novel_df = fetch_novel_data()
    novel_df = clean_novel_data(novel_df)
    novel_df = some_more_cleaning(novel_df)
    novel_df = drop_duplicate_rows(novel_df)
    novel_df = column_conversions(novel_df)
    novel_df = more_rows_and_inference_data('clean_1.csv')
    setting_expectations(novel_df)
    novel_df = extract_features(novel_df)
    novel_df = create_hit_miss(novel_df,quantile=0.35)
    send_to_feature_store(novel_df)

if __name__ == "__main__":
    main()

