use indexed_db_futures::{prelude::*, web_sys::DomException};
use uuid::Uuid;
use wasm_bindgen::JsValue;

use crate::user_profile::UserProfile;

pub async fn load_db() -> Result<IdbDatabase, DomException> {
    let mut db_req: OpenDbRequest = IdbDatabase::open_u32("wassimovie", 1)?;
    db_req.set_on_upgrade_needed(Some(|evt: &IdbVersionChangeEvent| -> Result<(), JsValue> {
        if !evt.db().object_store_names().any(|n| &n == "profiles") {
            evt.db().create_object_store("profiles")?;
        }
        Ok(())
    }));

    db_req.into_future().await
}

pub async fn get_profile(db: &IdbDatabase, prof_id: &str) -> Result<UserProfile, DomException> {
    let tx: IdbTransaction =
        db.transaction_on_one_with_mode("profiles", IdbTransactionMode::Readonly)?;
    let store: IdbObjectStore = tx.object_store("profiles")?;

    let value: Option<JsValue> = store.get_owned(prof_id.to_string())?.await?;
    let value = value.expect("Failed to load userprofile.");

    let profile: UserProfile =
        serde_wasm_bindgen::from_value(value).expect("Failed to convert profile");
    Ok(profile)
}

pub async fn save_profile(
    db: &IdbDatabase,
    prof_id: &str,
    prof: &UserProfile,
) -> Result<(), DomException> {
    let js_profile =
        serde_wasm_bindgen::to_value(prof).expect("Failed to convert profile to JS Object");
    let tx: IdbTransaction =
        db.transaction_on_one_with_mode("profiles", IdbTransactionMode::Readwrite)?;
    let store = tx.object_store("profiles")?;
    store.clear()?;

    store.put_key_val_owned(prof_id.to_string(), &js_profile)?;
    tx.await.into_result()
}
