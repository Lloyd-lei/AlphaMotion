"""Unified motion taxonomy: regex multi-label classifier shared by all sources.

classify(text) -> (classes:set, contacts:set, objects:set)
Primary class = first match in PRIORITY.
"""
import re

CLASSES = ["locomotion", "dynamic", "ground_contact", "sitting_on_object",
           "upper_body_manipulation", "whole_body_object", "gesture_social",
           "idle_posture", "other"]
PRIORITY = ["whole_body_object", "ground_contact", "sitting_on_object", "dynamic",
            "upper_body_manipulation", "locomotion", "gesture_social", "idle_posture", "other"]
CONTACTS = ["feet_only", "hands_ground", "knees", "hips", "back", "object_hands", "object_feet"]


def norm(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)          # camelCase split
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    s = s.lower().replace("don_t", "dont").replace("don't", "dont")
    s = re.sub(r"\b(\d+) ?cm box\b", r"\1cm platform", s)  # BONES climbing boxes -> platform
    s = re.sub(r"climbing box", "climbing platform", s)
    s = re.sub(r"[^a-z0-9/]+", " ", s)
    # neutralise phrases that would cause false positives
    s = re.sub(r"\blooking around on ground\b", "looking around low", s)
    s = re.sub(r"\bpush recovery\b|\bget(s|ting)? pushed\b|\bis pushed\b|\bpushed\b", " pushed_by_other ", s)
    s = re.sub(r"\bpush (and )?(fall|stumble)\b", " pushed_by_other fall ", s)
    return " " + re.sub(r"\s+", " ", s).strip() + " "


def R(p):
    return re.compile(r"(?<![a-z])(?:" + p + r")(?![a-z])")


LOCO = R(r"walk\w*|run|runs|running|runner|jog\w*|sprint\w*|strid\w*|stroll\w*|march\w*|limp\w*|"
         r"stairs?|upstairs|downstairs|staircase|slope\w*|incline|ramp|curb|backwards?|"
         r"sidestep\w*|side steps?|side stepping|sideways movement|forward movement|backwards movement|"
         r"straf\w*|shuffl\w*|tiptoe\w*|tip toe|sneak\w*|turn|turns|turning|turned|pivot\w*|"
         r"locomot\w*|moonwalk|catwalk|steps?|stepping|trot\w*|straight long|straight short|s shape\w*|"
         r"circle cc?w|clockwise|counterclockwise|arc cc?w|wander\w*|pace|pacing|navigat\w*|"
         r"walking straight|crouch walk\w*|crouchwalk|travel\w*|go forward|move forward|moves forward")
DYN = R(r"jump\w*|hop|hops|hopping|leap\w*|skip|skips|skipping|kick\w*|danc\w*|dancer|ballet|salsa|tango|"
        r"waltz|zumba|hip hop|hiphop|breakdanc\w*|jive|mambo|samba|cha cha|chacha|bachata|reggaeton|"
        r"macarena|choreograph\w*|krakowiak|kozak|vogue|martial\w*|karate|kung fu|taekwondo|capoeira|"
        r"boxing|boxer|shadow box\w*|punch\w*|jab|jabs|uppercut|hook punch|dodg\w*|evad\w*|evasion|"
        r"duck(s|ing)? (down|under|left|right)|fight\w*|combat|sword\w*|spar|sparring|strike\w*|"
        r"stab|stabs|stabbing|slash\w*|fall|falls|falling|fallen|stumbl\w*|trip|trips|tripped|tripping|slip|slips|"
        r"slipping|faint\w*|collaps\w*|get up|gets up|getting up|getup|get ups|rise up|rises up|"
        r"pushed_by_other|flip|flips|flipping|somersault\w*|cartwheel\w*|handspring\w*|acrobat\w*|"
        r"vault\w*|parkour|spin|spins|spinning|twirl\w*|pirouette\w*|jumping jacks?|burpee\w*|"
        r"tumbl\w*|sport\w*|play sport|sports move|hit|hits|hitting|attack\w*|stagger\w*|"
        r"aerial|tiger jump|monkey jump|swing kick|roundhouse|throw (a )?punch|block (a )?punch|"
        r"tennis|golf|baseball|badminton|volleyball|bowling|basketball|forehand|backhand|pitch(es|ing)?|"
        r"swing(s|ing)? (a |the )?(bat|racket|club|golf)")
GROUND = R(r"sit(s|ting)? (down )?on (the )?(floor|ground|heels|mat)|sitting on floor|sit on floor|"
           r"sitting on the floor|sitting on the ground|floor sit\w*|cross legged|sit cross\w*|legs crossed|"
           r"sitting legs|sit on heels|sitting on heels|kneel\w*|knelt|down on (his |her |their |one |both )?knees?|"
           r"knees? down|crawl\w*|all fours|hands and knees|lie|lies|lying|laid down|lay down|lays down|prone|"
           r"supine|push ?ups?|press ?ups?|plank\w*|handstand\w*|headstand\w*|hand stand\w*|hspu|"
           r"cartwheel\w*|burpee\w*|sit ?ups?|crunch\w*|roll(s|ing|ed)? (over|on|forward|backward|"
           r"backwards|sideways|to|around on)|(shoulder|safety|side|forward|backward|log) roll\w*|"
           r"somersault\w*|get(s|ting)? up from (the )?(floor|ground|lying)|"
           r"stand(s|ing)? up from (the )?(floor|ground|lying|kneel\w*|sitting on the (floor|ground))|"
           r"faint\w*|dead|death|dies|die|convulsion\w*|postmortem|ground (motion|1|2)|"
           r"(lie|lies|lying|sit|sits|sitting|sat|roll\w*|crawl\w*) (down )?(on|along|across) the (ground|floor)|yoga|touch(es|ing)? (the )?(ground|floor)|"
           r"hands? on (the )?(ground|floor)|touch ground|nailing floor|floor work|breakdanc\w*|"
           r"fall(s|ing)? (down )?(on|to|onto) (the )?(ground|floor|knees|back)|squat ?thrust|spider\w*|"
           r"bear crawl|lie down|lying down|on all fours|box dips?|dips|fall and get up")
SIT_OBJ = R(r"(sit\w*|seat\w*) (down )?(on|in|at|onto) (a |the |an )?(chair|bench|box|stool|table|couch|"
            r"sofa|bed|toilet|ledge|step|stairs|crate|desk|platform|seat)|chair|bench|stool|sofa|couch|"
            r"armchair|seated|sitting at (a |the )?table")
SIT_ANY = R(r"sit|sits|sitting|sat|sit down|sits down|sitting down")
UBM = R(r"pick\w*|place|places|placing|put|puts|putting|carr(y|ies|ying|ied)|push|pushes|pushing|pull|pulls|"
        r"pulling|hand over|handover|handing|give|gives|giving|given|pass|passes|passing|take|takes|taking|"
        r"took|grab|grabs|grabbing|grasp\w*|hold|holds|holding|lift|lifts|lifting|open|opens|opening|"
        r"close|closes|closing|door\w*|drawer\w*|cupboard\w*|fridge|microwave|knob|handle\w*|lever\w*|valve|"
        r"crank\w*|button|switch|tool\w*|hammer\w*|nail|nails|nailing|saw|saws|sawing|axe|chainsaw|"
        r"pickaxe|sledgehammer\w*|shovel\w*|dig|digs|digging|broom|mop|mops|mopping|sweep\w*|wip(e|es|ing)|"
        r"clean\w*|wash\w*|pour\w*|drink\w*|sip|eat|eats|eating|cut|cuts|cutting|chop\w*|stir\w*|whisk\w*|"
        r"grat(e|es|ing)|peel\w*|cook\w*|fry|frying|salt\w*|pepper|blender|knife|fork|spoon|plate|bowl|"
        r"mug|cup|cups|bottle|glass|wineglass|phone\w*|cellphone\w*|telephone|typ(e|es|ing)|keyboard|"
        r"writ(e|es|ing)|newspaper|book|guitar|violin|flute|recorder|piano|drum\w*|instrument\w*|"
        r"throw\w*|toss\w*|catch\w*|juggl\w*|ball|balls|object|objects|item|items|crate\w*|box|boxes|bag|"
        r"baby|sandbag|crutch\w*|cane|umbrella|watering|fishing|rope|lasso|binocular\w*|cigarette\w*|"
        r"smok\w*|lighter|apple|food|teapot|eyeglasses|stapler|mouse|camera|toothbrush|toothpaste|"
        r"flashlight|headphones|airplane|alarm ?clock|banana|cube\w*|cylinder\w*|sphere\w*|pyramid\w*|"
        r"torus\w*|stanford ?bunny|piggybank|scissors|water ?bottle|gamecontroller|game controller|"
        r"wristwatch|duck|elephant|lightbulb|offhand|inspect\w*|interact\w*|operat\w*|use object|"
        r"reach\w*|loot\w*|play(s|ing)? (with )?(a |the )?(cards|game)|card|cards|rifle|pistol|gun|"
        r"briefcase|shield|bat|racket|stick|weapon|spear|bow and arrow|feed\w*|egg|paint\w*|brush(es|ing)? teeth|"
        r"shower|wipe|towel|shav\w*|comb|combs|combing|dribbl\w*|tie|ties|tying|zip\w*|press something|move something")
WBO = R(r"(lift\w*|carr(y|ies|ying|ied)|pick\w* up|hold\w*|push\w*|pull\w*|drag\w*) "
        r"(\w+ ){0,4}(heavy|big|large|crate|box|boxes|cart|furniture|beam|barrel|sack|obstacle|"
        r"weight|weights|suitcase)|(?<!small )(?<!medium )heavy (object|item|box|crate|bag|thing)|big (heavy|light) (object|item|box)|"
        r"big heavy|medium big heavy|medium big light|big light|lift crate\w*|crate|"
        r"push(es|ing)? (a |the )?(cart|obstacle|car|wall|beam|table|box|heavy|chair)|push ?beam|push obstacle|"
        r"pull(s|ing)? (a |the )?(rope|cart|heavy|obstacle)|tug of war|kick(s|ing)? (a |the )?ball|"
        r"dribbl\w*|soccer\w*|football|basketball|shoot(s|ing)? (a |the )?(ball|basket)|"
        r"\d+cm platform|platform|climb\w*|ladder\w*|"
        r"jump(s|ing)? (on|onto|off|from) (a |the )?(box|platform|obstacle|table|bench|wall|\d+cm)|"
        r"step(s|ping)? (on|onto|up on|up onto) (a |the )?(box|platform|step|chair|bench)|"
        r"come (up|down) \d+cm|carry (a |the )?(person|partner)|wood log|lifting heavy|heavy lifting")
GEST = R(r"wav(e|es|ing)|point|points|pointing|poitning|talk\w*|speak\w*|convers\w*|chat\w*|gestur\w*|clap\w*|"
         r"cheer\w*|salut\w*|bow|bows|bowing|greet\w*|hello|goodbye|bye|shrug\w*|nod|nods|nodding|"
         r"thumbs? up|applau\w*|high five|hand shak\w*|handshake|shake hands|hug|hugs|hugging|threat\w*|"
         r"emot\w*|laugh\w*|cry|cries|crying|weep\w*|angry|anger|sad|happy|joy\w*|afraid|fear\w*|scare\w*|"
         r"bored|annoy\w*|confus\w*|surpris\w*|excit\w*|frustrat\w*|disappoint\w*|celebrat\w*|triumph\w*|"
         r"victory|pray\w*|beg|begs|begging|plead\w*|shout\w*|scream\w*|yell\w*|whisper\w*|beckon\w*|"
         r"sign language|signing|mime\w*|conduct\w*|kiss\w*|flirt\w*|omg|dont know|maybe|agree\w*|disagree\w*|"
         r"thinking|think|thinks|listen\w*|show bicep|flex\w*|crowd|quip|communicat\w*|acknowledg\w*|"
         r"apolog\w*|insult\w*|taunt\w*|mock\w*|lament\w*|despair\w*|worr\w*|nervous|shy|proud|pride|"
         r"sing|sings|singing|signal\w*|face ?palm|greeting|bye bye|come here|calling|call over|"
         r"satisfied|relieved|tired|thumb|stop gesture|"
         r"inflight safety|explain\w*|presentation|argu\w*|fist pump|dab|swear\w*|curs(e|ing)|insult")
IDLE = R(r"idle|idles|stand|stands|standing|still|t pose|a pose|tpose|apose|pose|poses|posing|stretch\w*|"
         r"warm ups?|warming up|adjust\w*|itch\w*|scratch\w*|relax\w*|wait\w*|rest|rests|resting|balanc\w*|"
         r"sway\w*|rom|range of motion|look|looks|looking|glanc\w*|breath\w*|shift(s|ing)? (weight|feet)|"
         r"weight shift\w*|body check|exercis\w*|squat\w*|lunge\w*|calisthenic\w*|tai chi|bend|bends|"
         r"bending|bent|lean\w*|crouch\w*|stoop\w*|fidget\w*|yawn\w*|sneez\w*|cough\w*|shiver\w*|"
         r"chicken wings|jiggl\w*|shake arms|shake hips|shake (his|her|their)? ?(arms|legs|hands)|"
         r"arm swing|swing(s|ing)? (his |her |their |both )?arms|rotat\w*|arm circles?|"
         r"raise(s|d)? (his |her |their |both |the |one |right |left )*(arms?|hands?|legs?|knees?)|"
         r"lower(s|ed)? (his |her |their |both |right |left )*(arms?|hands?)|"
         r"touch(es|ing)? (his |her |their )?(face|head|nose|body|knees|toes|shoulders)|brush\w*|"
         r"search\w*|calibrat\w*|arm movements|hand movements|leg movements|head movements|foot movements|"
         r"knee movement|waist movements|shoulder movements|upper body movements|wrist movements|"
         r"touching body part|raising body part|lowering body part|grab body part|touching face|stances?|"
         r"limits?|lar|uar|ulr|misc|static|army poses|motion range|stand still|"
         r"one leg loose|light hopping|hold position|stay|wiggl\w*|twist\w*|trembl\w*|freestyle")

CONTACT_RX = {
    "hands_ground": R(r"crawl\w*|all fours|hands and knees|push ?ups?|press ?ups?|plank\w*|handstand\w*|"
                      r"headstand\w*|hand stand\w*|hspu|cartwheel\w*|burpee\w*|spider\w*|bear crawl|"
                      r"hands? on (the )?(ground|floor)|touch(es|ing)? (the )?(ground|floor)|touch ground|"
                      r"get(s|ting)? up from (the )?(floor|ground|lying)|getup|get ups|fall and get up|roll(s|ing|ed)? (over|on|forward|backward|"
                      r"backwards|sideways|to)|(shoulder|safety|side|forward|backward|log) roll\w*|somersault\w*|"
                      r"breakdanc\w*|yoga|box dips?|dips|on hands|nailing floor|ground (motion|1|2)|"
                      r"stand(s|ing)? up from (the )?(floor|ground|lying)"),
    "knees": R(r"kneel\w*|knelt|down on (his |her |their |one |both )?knees?|knees? down|hands and knees|all fours|"
               r"crawl\w*|sit(ting)? on heels|nailing floor|proposal"),
    "hips": R(r"sit\w*|seat\w*|sat|lying|lie|lies|lay|lays|laying|laid|faint\w*|sit ?ups?|"
              r"fall(s|ing)? (down )?(on|to|onto) (the )?(ground|floor)|roll(s|ing|ed)? (over|on)|"
              r"ground (motion|1|2)|squat ?on|chair|bench|stool|sofa|couch"),
    "back": R(r"lying|lie|lies|lay|lays|laying|laid|supine|on (his |her |their )?back|sit ?ups?|crunch\w*|"
              r"(shoulder|safety|side|forward|backward|log) roll\w*|roll(s|ing|ed)? (over|on|forward|backward)|"
              r"somersault\w*|faint\w*|death|dead|convulsion\w*|postmortem|ground (motion|1|2)|"
              r"fall(s|ing)? (down )?(on|onto) (the |his |her |their )?back"),
    "object_feet": R(r"climb\w*|\d+cm platform|platform|step(s|ping)? (on|onto|up on)|stairs?|upstairs|"
                     r"downstairs|staircase|ladder\w*|kick(s|ing)? (a |the )?ball|dribbl\w*|soccer|football|"
                     r"on a box|on box|jump(s|ing)? (on|onto|off|from) (a |the )?(box|platform|\d+cm|table|bench)|"
                     r"edge|beam|curb|slope\w*|incline|ramp|stepping stones|handrail|come (up|down) \d+cm|"
                     r"move up down incline"),
}

OBJ_CATS = {
    "box_crate_generic": R(r"box|boxes|crate\w*|package|container|bin|basket|bag|object|objects|item|items|"
                           r"suitcase|briefcase|sack|barrel"),
    "furniture": R(r"chair|table|bench|stool|couch|sofa|bed|desk|shel(f|ves)|armchair"),
    "articulated_fixture": R(r"door\w*|drawer\w*|cupboard\w*|fridge|microwave|knob|handle\w*|lever\w*|valve|"
                             r"crank\w*|button|switch|window"),
    "tool": R(r"hammer\w*|nail|nails|nailing|saw|saws|sawing|axe|chainsaw|pickaxe|sledgehammer\w*|shovel\w*|"
              r"broom|mop|knife|mallet|screwdriver|wrench|drill|watering|fishing|rake|stapler|scissors|"
              r"sword\w*|rifle|pistol|gun|spear|shield|bow and arrow|weapon"),
    "kitchenware_food": R(r"mug|cup|cups|glass|wineglass|bottle|plate|bowl|spoon|fork|pan|frying|salt\w*|"
                          r"pepper|blender|dishes|apple|food|teapot|banana|egg|whisk\w*|meat|soup|vegetables?|"
                          r"bread|drink\w*|eat\w*"),
    "handheld_device": R(r"phone\w*|cellphone\w*|telephone|camera|newspaper|book|controller|gamecontroller|"
                         r"mouse|binocular\w*|flashlight|headphones|lighter|cigarette\w*|wristwatch|eyeglasses|"
                         r"toothbrush|toothpaste|keyboard|card|cards|flyers|purse|umbrella"),
    "instrument": R(r"guitar|violin|flute|recorder|piano|drum\w*|instrument\w*"),
    "ball_sport_gear": R(r"ball|balls|racket|punching bag|jump rope|rope|lasso|bat|golf|basketball|soccer|"
                         r"football|dribbl\w*"),
    "assistive": R(r"crutch\w*|cane|walker|wheelchair"),
    "person_or_doll": R(r"baby|sandbag|partner|carry (a |the )?person|another person|other person|grab person"),
    "terrain_env": R(r"wall|stairs?|upstairs|downstairs|ladder\w*|edge|obstacle\w*|pad|railing|handrail|"
                     r"\d+cm platform|platform|beam|slope\w*|ramp|incline|curb|scaffolding|tree"),
    "small_geometric_toy": R(r"cube\w*|cylinder\w*|sphere\w*|pyramid\w*|torus\w*|stanford ?bunny|elephant|duck|"
                             r"piggybank|alarm ?clock|airplane|lightbulb|toothpaste"),
}
_OBJ_NOUN_SRC = "|".join(["something|object|objects|item|items|thing|things|ball|box|boxes|bag|crate|chair|table|door|drawer|lid|bottle|cup|glass|phone|book|tool|stick|rope|weight|weights|bar|dumbbell|kettlebell|cart|basket|sponge|towel|pan|pot|plate|bowl|knife|spoon|food|drink|coffee|water|paper|pen|toy|shoe|shoes|hat|jacket|coat|shirt|bucket|bin|trash|can|jar|bread|fruit|apple|vegetable|tray|card|cards|lever|handle|knob|switch|button|wheel|steering|racket|bat|club|baby|sandbag"])
UBM_GENERIC = R(r"(put|puts|putting|place|places|placing|take|takes|taking|took|hold|holds|holding|push|pushes|"
                r"pushing|pull|pulls|pulling|open|opens|opening|close|closes|closing|pass|passes|passing|give|gives|"
                r"giving|lift|lifts|lifting|pick|picks|picking|grab|grabs|grabbing|move|moves|moving|reach|reaches|"
                r"reaching|carry|carries|carrying|throw|throws|throwing|drop|drops|dropping|set|sets|hand|hands)"
                r"( (up|down|out|off|over|into|onto|on|in|from|to|with|back|away|both|one|his|her|their|a|an|the|some|"
                r"two|small|large|big|heavy|light|right|left))* (" + _OBJ_NOUN_SRC + r")")
UBM_STRONG = R(r"pick(s|ed|ing)? up|grasp\w*|pour\w*|drink\w*|sip|eat|eats|eating|cut|cuts|cutting|chop\w*|"
               r"stir\w*|whisk\w*|grat(e|es|ing)|peel\w*|cook\w*|fry|frying|hammer\w*|nail|nails|nailing|saw|saws|"
               r"sawing|axe|chainsaw|pickaxe|sledgehammer\w*|shovel\w*|dig|digs|digging|broom|mop|mops|mopping|sweep\w*|"
               r"wip(e|es|ing)|clean\w*|wash\w*|throw\w*|toss\w*|catch\w*|juggl\w*|dribbl\w*|typ(e|es|ing)|keyboard|"
               r"writ(e|es|ing)|play(s|ing)? (the |a )?(guitar|violin|flute|recorder|piano|drums?|instrument)|"
               r"play instrument|guitar|violin|flute|recorder|piano|telephone|phone\w*|cellphone\w*|"
               r"interact with use object|touch object|grasp object|lift something|clean something|move something|"
               r"place something|take pick something up|give something|press something|open something|"
               r"close something|operate interface|action with ball|interact with rope|use object|"
               r"loot\w*|operat\w*|door\w*|drawer\w*|cupboard\w*|fridge|microwave|knob|lever\w*|valve|crank\w*|"
               r"button|watering|fishing|binocular\w*|cigarette\w*|smok\w*|lighter|teapot|eyeglasses|stapler|"
               r"toothbrush|toothpaste|flashlight|headphones|alarm ?clock|stanford ?bunny|piggybank|scissors|"
               r"gamecontroller|wristwatch|lightbulb|offhand|inspect\w*|shav\w*|comb|combs|combing|brush(es|ing)? teeth|"
               r"crate\w*|carr(y|ies|ying|ied)|ball|balls|rifle|pistol|gun|sword\w*|shield|racket|briefcase|"
               r"mug|bottle|wineglass|sponge|egg ?whisk|sandbag|baby|crutch\w*|newspaper|reach\w*|"
               r"feed\w*|paint\w*|sew\w*|knit\w*|tie (shoe|shoes|laces)|zip\w*|unzip\w*")

MANIP_OBJ_CATS = [k for k in OBJ_CATS if k not in ("terrain_env",)]


def classify(text: str, extra_classes=()):
    t = norm(text)
    cls = set(extra_classes)
    if LOCO.search(t): cls.add("locomotion")
    if DYN.search(t): cls.add("dynamic")
    g = bool(GROUND.search(t))
    if g: cls.add("ground_contact")
    if SIT_OBJ.search(t):
        cls.add("sitting_on_object")
    elif SIT_ANY.search(t) and not g:
        cls.add("sitting_on_object")          # ambiguous 'sit' with no floor cue -> chair/stool assumed
    if UBM_STRONG.search(t) or UBM_GENERIC.search(t): cls.add("upper_body_manipulation")
    if WBO.search(t): cls.add("whole_body_object")
    if GEST.search(t): cls.add("gesture_social")
    if IDLE.search(t): cls.add("idle_posture")
    if not cls: cls.add("other")
    objs = {k for k, rx in OBJ_CATS.items() if rx.search(t)}
    con = {k for k, rx in CONTACT_RX.items() if rx.search(t)}
    if ("sitting_on_object" in cls or "ground_contact" in cls) and SIT_ANY.search(t):
        con.add("hips")
    if {"upper_body_manipulation", "whole_body_object"} & cls and (objs & set(MANIP_OBJ_CATS)):
        con.add("object_hands")
    if not con:
        con.add("feet_only")
    return cls, con, objs


def primary(cls):
    for c in PRIORITY:
        if c in cls:
            return c
    return "other"
