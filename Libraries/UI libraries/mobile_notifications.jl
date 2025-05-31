// mobile_notifications.joyl - Cross-platform Notification System
pub struct NotificationCenter {
    platform_service: NotificationService,
    channels: HashMap<string, NotificationChannel>,
    pending: Vec<ScheduledNotification>,
    permission_state: PermissionState
}

impl NotificationCenter {
    /// Initialize notification system
    pub fn new() -> NotificationCenter {
        NotificationCenter {
            platform_service: PlatformNotificationService::new(),
            channels: HashMap::new(),
            pending: Vec::new(),
            permission_state: PermissionState::NotDetermined
        }
    }

    /// Request notification permissions
    pub async fn request_permissions(
        &mut self,
        options: PermissionOptions
    ) -> Result<PermissionState, NotificationError> {
        self.permission_state = self.platform_service
            .request_permission(options)
            .await?;
        
        Ok(self.permission_state)
    }

    /// Create notification channel (Android)
    pub fn create_channel(
        &mut self,
        id: string,
        name: string,
        importance: ChannelImportance
    ) -> Result<(), NotificationError> {
        let channel = NotificationChannel {
            id: id.clone(),
            name,
            importance,
            sound: None,
            vibration: true
        };
        
        self.platform_service.create_channel(&channel)?;
        self.channels.insert(id, channel);
        Ok(())
    }

    /// Schedule local notification
    pub fn schedule(
        &mut self,
        content: NotificationContent,
        trigger: NotificationTrigger
    ) -> Result<string, NotificationError> {
        let notification_id = generate_notification_id();
        let notification = PendingNotification {
            id: notification_id.clone(),
            content,
            trigger
        };
        
        self.platform_service.schedule(notification)?;
        self.pending.push(notification);
        
        Ok(notification_id)
    }
}