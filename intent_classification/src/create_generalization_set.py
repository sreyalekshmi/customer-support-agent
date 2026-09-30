"""
Phase 7: Generalization Test Set (hand-written)
Natural real-world phrasing, distinct from Bitext's templated style.
"""

import pandas as pd

generalization_data = [
    # cancel_order
    ("can you cancel the order i placed yesterday", "cancel_order"),
    ("i changed my mind, please cancel my purchase", "cancel_order"),
    ("how do i stop an order that hasn't shipped yet", "cancel_order"),
    ("cancel order 48291 right now", "cancel_order"),
    ("i don't want this order anymore, cancel it", "cancel_order"),

    # change_order
    ("i want to swap the color of the item i ordered", "change_order"),
    ("can i update the quantity on my order", "change_order"),
    ("need to change my order before it ships", "change_order"),
    ("is it possible to modify what's in my cart order", "change_order"),
    ("please change the size i selected on my last order", "change_order"),

    # change_shipping_address
    ("i moved, can you update my delivery address", "change_shipping_address"),
    ("wrong address on file, need to fix it", "change_shipping_address"),
    ("how do i change where my package gets sent", "change_shipping_address"),
    ("please ship to my new apartment instead", "change_shipping_address"),
    ("update my shipping details before it goes out", "change_shipping_address"),

    # check_cancellation_fee
    ("will i be charged anything if i cancel", "check_cancellation_fee"),
    ("is there a penalty for cancelling my order", "check_cancellation_fee"),
    ("what's the cancellation charge on this plan", "check_cancellation_fee"),
    ("do you charge a fee for cancelling early", "check_cancellation_fee"),
    ("how much does it cost to cancel", "check_cancellation_fee"),

    # check_invoice
    ("where can i see my invoice for last month", "check_invoice"),
    ("i need a copy of my billing invoice", "check_invoice"),
    ("can you show me the invoice for order 552", "check_invoice"),
    ("looking for my receipt from the recent purchase", "check_invoice"),
    ("how do i pull up my invoice history", "check_invoice"),

    # check_payment_methods
    ("what payment options do you accept", "check_payment_methods"),
    ("can i pay using paypal on your site", "check_payment_methods"),
    ("do you take debit cards", "check_payment_methods"),
    ("is cash on delivery available", "check_payment_methods"),
    ("which cards can i use to check out", "check_payment_methods"),

    # check_refund_policy
    ("what's your policy on refunds", "check_refund_policy"),
    ("how long do i have to request a refund", "check_refund_policy"),
    ("can i get my money back if i'm not happy", "check_refund_policy"),
    ("explain your return and refund rules", "check_refund_policy"),
    ("are refunds allowed on sale items", "check_refund_policy"),

    # complaint
    ("i'm really unhappy with the service i received", "complaint"),
    ("this is the third time my order has been late", "complaint"),
    ("i want to file a complaint about your support team", "complaint"),
    ("very disappointed with how this was handled", "complaint"),
    ("your app keeps crashing and it's frustrating", "complaint"),

    # contact_customer_service
    ("how do i get in touch with support", "contact_customer_service"),
    ("what's your customer service phone number", "contact_customer_service"),
    ("is there a live chat available", "contact_customer_service"),
    ("i need to reach out to your help desk", "contact_customer_service"),
    ("what are your support hours", "contact_customer_service"),

    # contact_human_agent
    ("i don't want a bot, connect me to a person", "contact_human_agent"),
    ("can i speak with a real human agent", "contact_human_agent"),
    ("transfer me to a live representative please", "contact_human_agent"),
    ("this bot isn't helping, get me a human", "contact_human_agent"),
    ("i need to talk to an actual agent", "contact_human_agent"),

    # create_account
    ("how do i sign up for a new account", "create_account"),
    ("i want to register on your platform", "create_account"),
    ("steps to create a profile with you", "create_account"),
    ("can you walk me through opening an account", "create_account"),
    ("i'd like to make a new user account", "create_account"),

    # delete_account
    ("please close my account permanently", "delete_account"),
    ("how do i deactivate my profile", "delete_account"),
    ("i want to delete all my account data", "delete_account"),
    ("remove my account from your system", "delete_account"),
    ("can you terminate my membership account", "delete_account"),

    # delivery_options
    ("what shipping methods do you offer", "delivery_options"),
    ("is express delivery available in my area", "delivery_options"),
    ("can i choose overnight shipping", "delivery_options"),
    ("what delivery speeds can i pick from", "delivery_options"),
    ("do you offer same-day delivery", "delivery_options"),

    # delivery_period
    ("how many days until my order arrives", "delivery_period"),
    ("what's the expected delivery time", "delivery_period"),
    ("when should i expect my package", "delivery_period"),
    ("how long does shipping usually take", "delivery_period"),
    ("estimated arrival date for my order", "delivery_period"),

    # edit_account
    ("i need to update my email on file", "edit_account"),
    ("how do i change my phone number in settings", "edit_account"),
    ("can i edit my profile information", "edit_account"),
    ("update my name on the account please", "edit_account"),
    ("how to change my account details", "edit_account"),

    # get_invoice
    ("send me the invoice for my last order", "get_invoice"),
    ("i need an invoice emailed to me", "get_invoice"),
    ("can you generate a receipt for this purchase", "get_invoice"),
    ("where do i download my invoice pdf", "get_invoice"),
    ("please issue an invoice for order 1123", "get_invoice"),

    # get_refund
    ("i'd like a refund for my damaged item", "get_refund"),
    ("please process my refund as soon as possible", "get_refund"),
    ("how do i request money back for this order", "get_refund"),
    ("i never received my item, refund me", "get_refund"),
    ("can you issue a refund to my card", "get_refund"),

    # newsletter_subscription
    ("how do i sign up for your email newsletter", "newsletter_subscription"),
    ("please unsubscribe me from your emails", "newsletter_subscription"),
    ("i want to stop getting promotional emails", "newsletter_subscription"),
    ("add me to your mailing list", "newsletter_subscription"),
    ("how to opt out of newsletter updates", "newsletter_subscription"),

    # payment_issue
    ("my card got declined at checkout", "payment_issue"),
    ("payment failed but money was deducted", "payment_issue"),
    ("i'm having trouble completing payment", "payment_issue"),
    ("why isn't my payment going through", "payment_issue"),
    ("there's an error when i try to pay", "payment_issue"),

    # place_order
    ("how do i place a new order", "place_order"),
    ("i want to buy this item, how do i order it", "place_order"),
    ("walk me through checking out", "place_order"),
    ("can you help me complete my purchase", "place_order"),
    ("i'd like to order this product now", "place_order"),

    # recover_password
    ("i forgot my password, how do i reset it", "recover_password"),
    ("can't log in, need to recover my account password", "recover_password"),
    ("send me a password reset link", "recover_password"),
    ("i'm locked out, help me reset my login", "recover_password"),
    ("how do i change a forgotten password", "recover_password"),

    # registration_problems
    ("i'm getting an error while signing up", "registration_problems"),
    ("registration form won't submit", "registration_problems"),
    ("can't verify my email during signup", "registration_problems"),
    ("having trouble creating an account, it keeps failing", "registration_problems"),
    ("signup process is stuck on verification", "registration_problems"),

    # review
    ("how do i leave a review for a product", "review"),
    ("i want to rate my recent purchase", "review"),
    ("can i edit a review i already posted", "review"),
    ("where do i submit feedback on an item", "review"),
    ("how to write a product rating", "review"),

    # set_up_shipping_address
    ("i haven't added a shipping address yet, how do i", "set_up_shipping_address"),
    ("help me set up my delivery address for the first time", "set_up_shipping_address"),
    ("how do i add a new address to my account", "set_up_shipping_address"),
    ("need to enter my shipping info before checkout", "set_up_shipping_address"),
    ("where do i input my address for deliveries", "set_up_shipping_address"),

    # switch_account
    ("how do i switch between my two accounts", "switch_account"),
    ("i want to change to a different user profile", "switch_account"),
    ("can i toggle to my business account", "switch_account"),
    ("switch my session to another account", "switch_account"),
    ("how to log into a different account", "switch_account"),

    # track_order
    ("where is my order right now", "track_order"),
    ("can you give me tracking info for my package", "track_order"),
    ("i want to see the status of my shipment", "track_order"),
    ("track my order number 7823", "track_order"),
    ("what's the current location of my delivery", "track_order"),

    # track_refund
    ("has my refund been processed yet", "track_refund"),
    ("checking the status of my refund request", "track_refund"),
    ("when will i see my refund in my account", "track_refund"),
    ("track refund for order 5521", "track_refund"),
    ("how much longer until my refund clears", "track_refund"),
]

df = pd.DataFrame(generalization_data, columns=["text", "true_intent"])
df.to_csv("data/generalization_test.csv", index=False)
print(f"Saved {len(df)} rows to data/generalization_test.csv")
print("Intents covered:", df['true_intent'].nunique())